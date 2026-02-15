"""
Model Replicator

Handles cross-region replication of ML models across AWS S3, GCP GCS, and Azure Blob Storage.
Supports versioning, integrity checks, and conflict resolution.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set
from pathlib import Path

import boto3
from google.cloud import storage as gcs_storage
from azure.storage.blob import BlobServiceClient
import aioboto3
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for ML models"""
    model_id: str
    version: str
    checksum: str
    size_bytes: int
    timestamp: str
    source_region: str
    target_regions: List[str]
    format: str  # pytorch, tensorflow, onnx, etc.
    framework_version: str
    tags: Dict[str, str]


@dataclass
class ReplicationStatus:
    """Status of a replication job"""
    model_id: str
    version: str
    source_region: str
    target_region: str
    status: str  # pending, in_progress, completed, failed
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    bytes_transferred: int


class CloudStorageAdapter:
    """Base adapter for cloud storage operations"""

    def __init__(self, region: str, provider: str):
        self.region = region
        self.provider = provider

    async def upload(self, source_path: str, destination: str) -> bool:
        raise NotImplementedError

    async def download(self, source: str, destination_path: str) -> bool:
        raise NotImplementedError

    async def list_objects(self, prefix: str) -> List[str]:
        raise NotImplementedError

    async def get_metadata(self, object_key: str) -> Dict:
        raise NotImplementedError

    async def delete(self, object_key: str) -> bool:
        raise NotImplementedError


class S3Adapter(CloudStorageAdapter):
    """AWS S3 storage adapter"""

    def __init__(self, region: str, bucket: str):
        super().__init__(region, "aws")
        self.bucket = bucket
        self.session = aioboto3.Session()

    async def upload(self, source_path: str, destination: str) -> bool:
        async with self.session.client('s3', region_name=self.region) as s3:
            try:
                await s3.upload_file(source_path, self.bucket, destination)
                logger.info(f"Uploaded {source_path} to s3://{self.bucket}/{destination}")
                return True
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                return False

    async def download(self, source: str, destination_path: str) -> bool:
        async with self.session.client('s3', region_name=self.region) as s3:
            try:
                await s3.download_file(self.bucket, source, destination_path)
                logger.info(f"Downloaded s3://{self.bucket}/{source} to {destination_path}")
                return True
            except Exception as e:
                logger.error(f"S3 download failed: {e}")
                return False

    async def list_objects(self, prefix: str) -> List[str]:
        async with self.session.client('s3', region_name=self.region) as s3:
            paginator = s3.get_paginator('list_objects_v2')
            objects = []
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if 'Contents' in page:
                    objects.extend([obj['Key'] for obj in page['Contents']])
            return objects

    async def get_metadata(self, object_key: str) -> Dict:
        async with self.session.client('s3', region_name=self.region) as s3:
            response = await s3.head_object(Bucket=self.bucket, Key=object_key)
            return {
                'size': response['ContentLength'],
                'etag': response['ETag'].strip('"'),
                'last_modified': response['LastModified'].isoformat(),
                'metadata': response.get('Metadata', {})
            }

    async def delete(self, object_key: str) -> bool:
        async with self.session.client('s3', region_name=self.region) as s3:
            try:
                await s3.delete_object(Bucket=self.bucket, Key=object_key)
                return True
            except Exception as e:
                logger.error(f"S3 delete failed: {e}")
                return False


class GCSAdapter(CloudStorageAdapter):
    """GCP Cloud Storage adapter"""

    def __init__(self, region: str, bucket: str):
        super().__init__(region, "gcp")
        self.bucket_name = bucket
        self.client = gcs_storage.Client()
        self.bucket = self.client.bucket(bucket)

    async def upload(self, source_path: str, destination: str) -> bool:
        try:
            blob = self.bucket.blob(destination)
            await asyncio.to_thread(blob.upload_from_filename, source_path)
            logger.info(f"Uploaded {source_path} to gs://{self.bucket_name}/{destination}")
            return True
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            return False

    async def download(self, source: str, destination_path: str) -> bool:
        try:
            blob = self.bucket.blob(source)
            await asyncio.to_thread(blob.download_to_filename, destination_path)
            logger.info(f"Downloaded gs://{self.bucket_name}/{source} to {destination_path}")
            return True
        except Exception as e:
            logger.error(f"GCS download failed: {e}")
            return False

    async def list_objects(self, prefix: str) -> List[str]:
        blobs = await asyncio.to_thread(
            lambda: list(self.client.list_blobs(self.bucket_name, prefix=prefix))
        )
        return [blob.name for blob in blobs]

    async def get_metadata(self, object_key: str) -> Dict:
        blob = self.bucket.blob(object_key)
        await asyncio.to_thread(blob.reload)
        return {
            'size': blob.size,
            'etag': blob.etag,
            'last_modified': blob.updated.isoformat() if blob.updated else None,
            'metadata': blob.metadata or {}
        }

    async def delete(self, object_key: str) -> bool:
        try:
            blob = self.bucket.blob(object_key)
            await asyncio.to_thread(blob.delete)
            return True
        except Exception as e:
            logger.error(f"GCS delete failed: {e}")
            return False


class AzureBlobAdapter(CloudStorageAdapter):
    """Azure Blob Storage adapter"""

    def __init__(self, region: str, connection_string: str, container: str):
        super().__init__(region, "azure")
        self.container = container
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service.get_container_client(container)

    async def upload(self, source_path: str, destination: str) -> bool:
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container, blob=destination
            )
            with open(source_path, 'rb') as data:
                await asyncio.to_thread(blob_client.upload_blob, data, overwrite=True)
            logger.info(f"Uploaded {source_path} to azure://{self.container}/{destination}")
            return True
        except Exception as e:
            logger.error(f"Azure upload failed: {e}")
            return False

    async def download(self, source: str, destination_path: str) -> bool:
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container, blob=source
            )
            with open(destination_path, 'wb') as file:
                data = await asyncio.to_thread(blob_client.download_blob)
                await asyncio.to_thread(file.write, data.readall())
            logger.info(f"Downloaded azure://{self.container}/{source} to {destination_path}")
            return True
        except Exception as e:
            logger.error(f"Azure download failed: {e}")
            return False

    async def list_objects(self, prefix: str) -> List[str]:
        blobs = await asyncio.to_thread(
            lambda: list(self.container_client.list_blobs(name_starts_with=prefix))
        )
        return [blob.name for blob in blobs]

    async def get_metadata(self, object_key: str) -> Dict:
        blob_client = self.blob_service.get_blob_client(
            container=self.container, blob=object_key
        )
        properties = await asyncio.to_thread(blob_client.get_blob_properties)
        return {
            'size': properties.size,
            'etag': properties.etag,
            'last_modified': properties.last_modified.isoformat(),
            'metadata': properties.metadata or {}
        }

    async def delete(self, object_key: str) -> bool:
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container, blob=object_key
            )
            await asyncio.to_thread(blob_client.delete_blob)
            return True
        except Exception as e:
            logger.error(f"Azure delete failed: {e}")
            return False


class ModelReplicator:
    """
    Multi-region ML model replicator

    Manages replication of ML models across AWS, GCP, and Azure regions
    with integrity checking, versioning, and conflict resolution.
    """

    def __init__(self, config: Dict, registry=None):
        self.config = config
        self.adapters: Dict[str, CloudStorageAdapter] = {}
        self.replication_queue: asyncio.Queue = asyncio.Queue()
        self.status_tracker: Dict[str, ReplicationStatus] = {}
        self._initialize_adapters()
        
        # Metrics
        from prometheus_client import Counter, Histogram
        self.registry = registry
        if self.registry:
            self.replication_counter = Counter(
                'model_replication_total', 
                'Total model replication attempts',
                ['source_region', 'target_region', 'status'],
                registry=self.registry
            )
            self.replication_duration = Histogram(
                'model_replication_duration_seconds',
                'Model replication duration',
                ['source_region', 'target_region'],
                registry=self.registry
            )

    def _initialize_adapters(self):
        """Initialize storage adapters for each region"""
        for region_config in self.config.get('regions', []):
            region_name = region_config['name']
            provider = region_config['provider']

            if provider == 'aws':
                adapter = S3Adapter(
                    region=region_config['aws_region'],
                    bucket=region_config['bucket']
                )
            elif provider == 'gcp':
                adapter = GCSAdapter(
                    region=region_config['gcp_region'],
                    bucket=region_config['bucket']
                )
            elif provider == 'azure':
                adapter = AzureBlobAdapter(
                    region=region_config['azure_region'],
                    connection_string=region_config['connection_string'],
                    container=region_config['container']
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            self.adapters[region_name] = adapter
            logger.info(f"Initialized {provider} adapter for {region_name}")

    @staticmethod
    async def compute_checksum(file_path: str) -> str:
        """Compute SHA256 checksum of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def register_model(
        self,
        model_path: str,
        model_id: str,
        version: str,
        source_region: str,
        target_regions: List[str],
        format: str = "pytorch",
        framework_version: str = "2.0.0",
        tags: Optional[Dict[str, str]] = None
    ) -> ModelMetadata:
        """Register a new model for replication"""

        checksum = await self.compute_checksum(model_path)
        size_bytes = Path(model_path).stat().st_size

        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            checksum=checksum,
            size_bytes=size_bytes,
            timestamp=datetime.utcnow().isoformat(),
            source_region=source_region,
            target_regions=target_regions,
            format=format,
            framework_version=framework_version,
            tags=tags or {}
        )

        logger.info(f"Registered model {model_id} v{version} for replication")
        return metadata

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def replicate_model(
        self,
        model_path: str,
        metadata: ModelMetadata
    ) -> Dict[str, ReplicationStatus]:
        """Replicate model from source to target regions"""

        results = {}
        source_adapter = self.adapters[metadata.source_region]

        # Upload to source region first
        source_key = f"models/{metadata.model_id}/{metadata.version}/model.bin"
        await source_adapter.upload(model_path, source_key)

        # Upload metadata
        metadata_key = f"models/{metadata.model_id}/{metadata.version}/metadata.json"
        metadata_path = f"/tmp/{metadata.model_id}_{metadata.version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f)
        await source_adapter.upload(metadata_path, metadata_key)

        # Replicate to target regions
        tasks = []
        for target_region in metadata.target_regions:
            if target_region != metadata.source_region:
                task = self._replicate_to_region(
                    source_key, metadata_key, metadata, target_region
                )
                tasks.append(task)

        replication_results = await asyncio.gather(*tasks, return_exceptions=True)

        for target_region, result in zip(metadata.target_regions, replication_results):
            if isinstance(result, Exception):
                status = ReplicationStatus(
                    model_id=metadata.model_id,
                    version=metadata.version,
                    source_region=metadata.source_region,
                    target_region=target_region,
                    status="failed",
                    started_at=datetime.utcnow().isoformat(),
                    completed_at=datetime.utcnow().isoformat(),
                    error_message=str(result),
                    bytes_transferred=0
                )
            else:
                status = result

            results[target_region] = status
            self.status_tracker[f"{metadata.model_id}:{target_region}"] = status

        return results

    async def _replicate_to_region(
        self,
        source_key: str,
        metadata_key: str,
        metadata: ModelMetadata,
        target_region: str
    ) -> ReplicationStatus:
        """Replicate model and metadata to a specific region"""

        status = ReplicationStatus(
            model_id=metadata.model_id,
            version=metadata.version,
            source_region=metadata.source_region,
            target_region=target_region,
            status="in_progress",
            started_at=datetime.utcnow().isoformat(),
            completed_at=None,
            error_message=None,
            bytes_transferred=0
        )

        try:
            source_adapter = self.adapters[metadata.source_region]
            target_adapter = self.adapters[target_region]

            # Download from source
            temp_model_path = f"/tmp/{metadata.model_id}_{metadata.version}_model.bin"
            temp_metadata_path = f"/tmp/{metadata.model_id}_{metadata.version}_metadata.json"

            await source_adapter.download(source_key, temp_model_path)
            await source_adapter.download(metadata_key, temp_metadata_path)

            # Verify checksum
            downloaded_checksum = await self.compute_checksum(temp_model_path)
            if downloaded_checksum != metadata.checksum:
                raise ValueError("Checksum mismatch after download")

            # Upload to target
            await target_adapter.upload(temp_model_path, source_key)
            await target_adapter.upload(temp_metadata_path, metadata_key)

            # Verify upload
            target_metadata = await target_adapter.get_metadata(source_key)
            if target_metadata['size'] != metadata.size_bytes:
                raise ValueError("Size mismatch in target region")

            status.status = "completed"
            status.completed_at = datetime.utcnow().isoformat()
            status.bytes_transferred = metadata.size_bytes

            logger.info(
                f"Successfully replicated {metadata.model_id} v{metadata.version} "
                f"to {target_region}"
            )

            # Record metrics
            if self.registry:
                self.replication_counter.labels(
                    source_region=metadata.source_region,
                    target_region=target_region,
                    status="success"
                ).inc()
                
                duration = (datetime.utcnow() - datetime.fromisoformat(status.started_at)).total_seconds()
                self.replication_duration.labels(
                    source_region=metadata.source_region,
                    target_region=target_region
                ).observe(duration)

            # Cleanup temp files
            Path(temp_model_path).unlink(missing_ok=True)
            Path(temp_metadata_path).unlink(missing_ok=True)

        except Exception as e:
            status.status = "failed"
            status.error_message = str(e)
            status.completed_at = datetime.utcnow().isoformat()
            logger.error(f"Replication to {target_region} failed: {e}")
            
            # Record metrics
            if self.registry:
                self.replication_counter.labels(
                    source_region=metadata.source_region,
                    target_region=target_region,
                    status="failure"
                ).inc()

        return status

    async def sync_models(self) -> Dict[str, List[str]]:
        """Synchronize all models across regions"""
        all_models = {}

        # Discover models in each region
        for region_name, adapter in self.adapters.items():
            models = await adapter.list_objects("models/")
            # Filter to only include model binaries, not metadata
            model_binaries = [m for m in models if m.endswith('model.bin')]
            all_models[region_name] = model_binaries
            logger.info(f"Found {len(model_binaries)} models in {region_name}")

        # Find missing models in each region
        # We assume models are keyed by "models/{model_id}/{version}/model.bin"
        unique_model_keys = set()
        for models in all_models.values():
            unique_model_keys.update(models)

        sync_plan = {}
        for region_name, models in all_models.items():
            existing_set = set(models)
            missing = []
            for key in unique_model_keys:
                if key not in existing_set:
                    missing.append(key)
            
            if missing:
                sync_plan[region_name] = missing
                logger.info(f"Region {region_name} missing {len(missing)} models")

        # Execute Sync
        for target_region, missing_keys in sync_plan.items():
            for key in missing_keys:
                # Key format: models/{model_id}/{version}/model.bin
                try:
                    parts = key.split('/')
                    if len(parts) >= 4:
                        model_id = parts[1]
                        version = parts[2]
                        
                        # Find a source region that has this model
                        source_region = None
                        for r, m in all_models.items():
                            if key in m:
                                source_region = r
                                break
                        
                        if source_region:
                            # Fetch metadata first (we need it for Replicate call)
                            metadata_key = f"models/{model_id}/{version}/metadata.json"
                            source_adapter = self.adapters[source_region]
                            try:
                                # We need to download metadata to construct ModelMetadata object
                                # or we can implement a lighter weight copy if adapters support it.
                                # For now, let's respect the existing replicate_model signature
                                # which handles download/upload.
                                # But replicate_model expects a LOCAL file path.
                                # We need a cross-region copy method.
                                
                                # Let's use _replicate_to_region directly if we can construct metadata
                                # Download metadata first
                                temp_meta_path = f"/tmp/{model_id}_{version}_meta_sync.json"
                                await source_adapter.download(metadata_key, temp_meta_path)
                                
                                with open(temp_meta_path, 'r') as f:
                                    meta_dict = json.load(f)
                                
                                metadata = ModelMetadata(**meta_dict)
                                
                                # Trigger replication
                                await self._replicate_to_region(
                                    key, metadata_key, metadata, target_region
                                )
                                
                                Path(temp_meta_path).unlink(missing_ok=True)
                                
                            except Exception as e:
                                logger.error(f"Failed to sync {key} from {source_region} to {target_region}: {e}")
                        else:
                            logger.error(f"Source not found for {key}")
                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")

        return sync_plan

    async def continuous_replication(self, interval_seconds: int = 300):
        """Continuously replicate models across regions"""
        logger.info(f"Starting continuous replication (interval: {interval_seconds}s)")
        while True:
            try:
                await self.sync_models()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in continuous replication: {e}")
                await asyncio.sleep(interval_seconds)

    async def get_replication_status(
        self,
        model_id: str,
        target_region: Optional[str] = None
    ) -> Dict[str, ReplicationStatus]:
        """Get replication status for a model"""

        if target_region:
            key = f"{model_id}:{target_region}"
            return {target_region: self.status_tracker.get(key)}

        # Return all statuses for this model
        results = {}
        for key, status in self.status_tracker.items():
            if key.startswith(f"{model_id}:"):
                results[status.target_region] = status

        return results
