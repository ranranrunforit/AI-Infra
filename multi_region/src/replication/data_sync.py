"""
Data Synchronization Service

Handles cross-region data synchronization for training data, validation datasets,
and feature stores across multiple cloud providers.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from enum import Enum

import boto3
from google.cloud import storage as gcs_storage
from azure.storage.blob import BlobServiceClient
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyncStrategy(Enum):
    """Data synchronization strategies"""
    FULL = "full"  # Full sync - copy everything
    INCREMENTAL = "incremental"  # Only sync changes
    BIDIRECTIONAL = "bidirectional"  # Two-way sync
    MASTER_REPLICA = "master_replica"  # One-way from master


@dataclass
class DatasetMetadata:
    """Metadata for datasets"""
    dataset_id: str
    name: str
    version: str
    size_bytes: int
    file_count: int
    checksum: str
    created_at: str
    updated_at: str
    source_region: str
    tags: Dict[str, str]
    schema_version: str


@dataclass
class SyncJob:
    """Represents a data synchronization job"""
    job_id: str
    dataset_id: str
    source_region: str
    target_regions: List[str]
    strategy: SyncStrategy
    status: str
    progress_percent: float
    files_synced: int
    files_total: int
    bytes_transferred: int
    started_at: str
    completed_at: Optional[str]
    error_message: Optional[str]


@dataclass
class SyncConflict:
    """Represents a synchronization conflict"""
    dataset_id: str
    file_path: str
    source_region: str
    target_region: str
    source_checksum: str
    target_checksum: str
    source_modified: str
    target_modified: str
    resolution: Optional[str]  # "source_wins", "target_wins", "manual"


class DataSync:
    """
    Cross-region data synchronization service

    Manages synchronization of datasets across multiple cloud regions
    with conflict detection and resolution.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.sync_jobs: Dict[str, SyncJob] = {}
        self.conflicts: List[SyncConflict] = []
        self.sync_interval = config.get('sync_interval_seconds', 300)
        self._running = False

        # Storage adapters (reuse from model_replicator)
        from .model_replicator import S3Adapter, GCSAdapter, AzureBlobAdapter
        self.adapters: Dict[str, any] = {}
        self._initialize_adapters()

    def _initialize_adapters(self):
        """Initialize storage adapters for each region"""
        from .model_replicator import S3Adapter, GCSAdapter, AzureBlobAdapter

        for region_config in self.config.get('regions', []):
            region_name = region_config['name']
            provider = region_config['provider']

            if provider == 'aws':
                adapter = S3Adapter(
                    region=region_config['aws_region'],
                    bucket=region_config['data_bucket']
                )
            elif provider == 'gcp':
                adapter = GCSAdapter(
                    region=region_config['gcp_region'],
                    bucket=region_config['data_bucket']
                )
            elif provider == 'azure':
                adapter = AzureBlobAdapter(
                    region=region_config['azure_region'],
                    connection_string=region_config['connection_string'],
                    container=region_config['data_container']
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            self.adapters[region_name] = adapter
            logger.info(f"Initialized adapter for {region_name}")

    @staticmethod
    async def compute_checksum(file_path: str) -> str:
        """Compute SHA256 checksum"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def register_dataset(
        self,
        dataset_path: str,
        dataset_id: str,
        name: str,
        version: str,
        source_region: str,
        tags: Optional[Dict[str, str]] = None
    ) -> DatasetMetadata:
        """Register a new dataset for synchronization"""

        # Calculate dataset metrics
        dataset_path_obj = Path(dataset_path)
        if dataset_path_obj.is_dir():
            files = list(dataset_path_obj.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            size_bytes = sum(f.stat().st_size for f in files if f.is_file())
        else:
            file_count = 1
            size_bytes = dataset_path_obj.stat().st_size

        # Compute aggregate checksum
        checksums = []
        if dataset_path_obj.is_dir():
            for f in sorted(dataset_path_obj.rglob('*')):
                if f.is_file():
                    checksums.append(await self.compute_checksum(str(f)))
        else:
            checksums.append(await self.compute_checksum(dataset_path))

        aggregate_checksum = hashlib.sha256(
            ''.join(checksums).encode()
        ).hexdigest()

        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            version=version,
            size_bytes=size_bytes,
            file_count=file_count,
            checksum=aggregate_checksum,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            source_region=source_region,
            tags=tags or {},
            schema_version="1.0"
        )

        logger.info(f"Registered dataset {dataset_id} v{version}")
        return metadata

    async def create_sync_job(
        self,
        dataset_metadata: DatasetMetadata,
        target_regions: List[str],
        strategy: SyncStrategy = SyncStrategy.INCREMENTAL
    ) -> SyncJob:
        """Create a new synchronization job"""

        job_id = f"sync-{dataset_metadata.dataset_id}-{datetime.utcnow().timestamp()}"

        job = SyncJob(
            job_id=job_id,
            dataset_id=dataset_metadata.dataset_id,
            source_region=dataset_metadata.source_region,
            target_regions=target_regions,
            strategy=strategy,
            status="pending",
            progress_percent=0.0,
            files_synced=0,
            files_total=dataset_metadata.file_count,
            bytes_transferred=0,
            started_at=datetime.utcnow().isoformat(),
            completed_at=None,
            error_message=None
        )

        self.sync_jobs[job_id] = job
        logger.info(f"Created sync job {job_id} for {dataset_metadata.dataset_id}")
        return job

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def execute_sync_job(self, job: SyncJob) -> SyncJob:
        """Execute a synchronization job"""

        job.status = "in_progress"
        source_adapter = self.adapters[job.source_region]

        try:
            # List all files in source dataset
            prefix = f"datasets/{job.dataset_id}/"
            source_files = await source_adapter.list_objects(prefix)
            job.files_total = len(source_files)

            logger.info(f"Syncing {len(source_files)} files to {len(job.target_regions)} regions")

            # Sync to each target region
            for target_region in job.target_regions:
                if target_region == job.source_region:
                    continue

                target_adapter = self.adapters[target_region]

                # Get list of files in target
                target_files_set = set(await target_adapter.list_objects(prefix))

                # Determine files to sync based on strategy
                files_to_sync = []

                if job.strategy == SyncStrategy.FULL:
                    files_to_sync = source_files
                elif job.strategy == SyncStrategy.INCREMENTAL:
                    # Only sync new or modified files
                    for file_key in source_files:
                        if file_key not in target_files_set:
                            files_to_sync.append(file_key)
                        else:
                            # Check if file has been modified
                            source_meta = await source_adapter.get_metadata(file_key)
                            target_meta = await target_adapter.get_metadata(file_key)

                            if source_meta['size'] != target_meta['size']:
                                files_to_sync.append(file_key)
                            elif source_meta.get('etag') != target_meta.get('etag'):
                                # Potential conflict - need manual resolution
                                conflict = SyncConflict(
                                    dataset_id=job.dataset_id,
                                    file_path=file_key,
                                    source_region=job.source_region,
                                    target_region=target_region,
                                    source_checksum=source_meta.get('etag', ''),
                                    target_checksum=target_meta.get('etag', ''),
                                    source_modified=source_meta.get('last_modified', ''),
                                    target_modified=target_meta.get('last_modified', ''),
                                    resolution=None
                                )
                                self.conflicts.append(conflict)
                                logger.warning(f"Conflict detected: {file_key}")

                # Execute file transfers
                for file_key in files_to_sync:
                    # Download from source
                    temp_path = f"/tmp/{Path(file_key).name}"
                    await source_adapter.download(file_key, temp_path)

                    # Upload to target
                    await target_adapter.upload(temp_path, file_key)

                    # Update progress
                    job.files_synced += 1
                    job.progress_percent = (job.files_synced / job.files_total) * 100

                    # Cleanup
                    Path(temp_path).unlink(missing_ok=True)

                logger.info(f"Synced {len(files_to_sync)} files to {target_region}")

            job.status = "completed"
            job.completed_at = datetime.utcnow().isoformat()
            job.progress_percent = 100.0

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow().isoformat()
            logger.error(f"Sync job {job.job_id} failed: {e}")

        return job

    async def resolve_conflict(
        self,
        conflict: SyncConflict,
        resolution: str
    ) -> bool:
        """Resolve a synchronization conflict"""

        if resolution not in ["source_wins", "target_wins", "manual"]:
            raise ValueError("Invalid resolution strategy")

        conflict.resolution = resolution
        logger.info(f"Resolved conflict for {conflict.file_path}: {resolution}")

        if resolution == "source_wins":
            # Copy from source to target
            source_adapter = self.adapters[conflict.source_region]
            target_adapter = self.adapters[conflict.target_region]

            temp_path = f"/tmp/{Path(conflict.file_path).name}"
            await source_adapter.download(conflict.file_path, temp_path)
            await target_adapter.upload(temp_path, conflict.file_path)
            Path(temp_path).unlink(missing_ok=True)

        elif resolution == "target_wins":
            # Keep target version, log decision
            logger.info(f"Keeping target version for {conflict.file_path}")

        return True

    async def continuous_sync(self):
        """Continuously synchronize data across regions"""

        self._running = True
        logger.info(f"Starting continuous sync with interval {self.sync_interval}s")

        while self._running:
            try:
                # Discover datasets in all regions
                for region_name, adapter in self.adapters.items():
                    datasets = await adapter.list_objects("datasets/")

                    # Create sync jobs for new datasets
                    for dataset_key in datasets:
                        # Check if this dataset needs syncing
                        # This is a simplified version - in production,
                        # you'd track which datasets have been synced
                        pass

                await asyncio.sleep(self.sync_interval)

            except Exception as e:
                logger.error(f"Error in continuous sync: {e}")
                await asyncio.sleep(self.sync_interval)

    def stop_continuous_sync(self):
        """Stop continuous synchronization"""
        self._running = False
        logger.info("Stopped continuous sync")

    async def get_sync_status(self, job_id: str) -> Optional[SyncJob]:
        """Get status of a sync job"""
        return self.sync_jobs.get(job_id)

    async def get_conflicts(
        self,
        dataset_id: Optional[str] = None
    ) -> List[SyncConflict]:
        """Get all unresolved conflicts"""

        if dataset_id:
            return [c for c in self.conflicts if c.dataset_id == dataset_id and not c.resolution]

        return [c for c in self.conflicts if not c.resolution]

    async def verify_sync(
        self,
        dataset_id: str,
        source_region: str,
        target_region: str
    ) -> Dict:
        """Verify that a dataset is properly synced"""

        source_adapter = self.adapters[source_region]
        target_adapter = self.adapters[target_region]

        prefix = f"datasets/{dataset_id}/"

        source_files = set(await source_adapter.list_objects(prefix))
        target_files = set(await target_adapter.list_objects(prefix))

        missing_in_target = source_files - target_files
        extra_in_target = target_files - source_files

        # Check file integrity for common files
        common_files = source_files.intersection(target_files)
        mismatched_files = []

        for file_key in list(common_files)[:100]:  # Sample first 100 files
            source_meta = await source_adapter.get_metadata(file_key)
            target_meta = await target_adapter.get_metadata(file_key)

            if source_meta['size'] != target_meta['size']:
                mismatched_files.append({
                    'file': file_key,
                    'source_size': source_meta['size'],
                    'target_size': target_meta['size']
                })

        result = {
            'dataset_id': dataset_id,
            'source_region': source_region,
            'target_region': target_region,
            'in_sync': len(missing_in_target) == 0 and len(extra_in_target) == 0 and len(mismatched_files) == 0,
            'source_files': len(source_files),
            'target_files': len(target_files),
            'missing_in_target': len(missing_in_target),
            'extra_in_target': len(extra_in_target),
            'mismatched_files': len(mismatched_files),
            'sample_mismatches': mismatched_files[:10]
        }

        logger.info(f"Sync verification: {result}")
        return result

    async def get_replication_lag(
        self,
        dataset_id: str
    ) -> Dict[Tuple[str, str], timedelta]:
        """Calculate replication lag between regions"""

        lags = {}

        for source_region in self.adapters.keys():
            source_adapter = self.adapters[source_region]
            metadata_key = f"datasets/{dataset_id}/metadata.json"

            try:
                source_meta = await source_adapter.get_metadata(metadata_key)
                source_time = datetime.fromisoformat(
                    source_meta.get('last_modified', '').replace('Z', '+00:00')
                )

                for target_region in self.adapters.keys():
                    if target_region == source_region:
                        continue

                    target_adapter = self.adapters[target_region]
                    target_meta = await target_adapter.get_metadata(metadata_key)
                    target_time = datetime.fromisoformat(
                        target_meta.get('last_modified', '').replace('Z', '+00:00')
                    )

                    lag = abs(source_time - target_time)
                    lags[(source_region, target_region)] = lag

            except Exception as e:
                logger.warning(f"Could not calculate lag for {source_region}: {e}")

        return lags
