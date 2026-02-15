"""
Configuration Synchronization Service

Handles cross-region synchronization of configuration data including:
- Application configs
- Feature flags
- Model serving configs
- Infrastructure configurations
"""

import asyncio
import hashlib
import json
import logging
import yaml
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigType(Enum):
    """Types of configuration"""
    APPLICATION = "application"
    FEATURE_FLAG = "feature_flag"
    MODEL_SERVING = "model_serving"
    INFRASTRUCTURE = "infrastructure"
    SECRETS = "secrets"


@dataclass
class ConfigItem:
    """Represents a configuration item"""
    config_id: str
    name: str
    config_type: ConfigType
    data: Dict[str, Any]
    version: str
    checksum: str
    created_at: str
    updated_at: str
    source_region: str
    encrypted: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConfigSyncStatus:
    """Status of configuration synchronization"""
    config_id: str
    source_region: str
    target_region: str
    status: str  # synced, out_of_sync, error
    last_synced_at: Optional[str]
    version: str
    error_message: Optional[str]


class ConfigurationStore:
    """Base class for configuration storage backends"""

    async def get(self, key: str) -> Optional[Dict]:
        raise NotImplementedError

    async def set(self, key: str, value: Dict, version: str) -> bool:
        raise NotImplementedError

    async def list(self, prefix: str) -> List[str]:
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        raise NotImplementedError

    async def get_version(self, key: str) -> Optional[str]:
        raise NotImplementedError


class DynamoDBConfigStore(ConfigurationStore):
    """DynamoDB-based configuration store"""

    def __init__(self, table_name: str, region: str):
        self.table_name = table_name
        self.region = region
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)

    async def get(self, key: str) -> Optional[Dict]:
        try:
            response = await asyncio.to_thread(
                self.table.get_item,
                Key={'config_id': key}
            )
            return response.get('Item')
        except ClientError as e:
            logger.error(f"DynamoDB get failed: {e}")
            return None

    async def set(self, key: str, value: Dict, version: str) -> bool:
        try:
            item = {
                'config_id': key,
                'data': value,
                'version': version,
                'updated_at': datetime.utcnow().isoformat()
            }
            await asyncio.to_thread(
                self.table.put_item,
                Item=item
            )
            logger.info(f"Saved config {key} v{version} to DynamoDB")
            return True
        except ClientError as e:
            logger.error(f"DynamoDB put failed: {e}")
            return False

    async def list(self, prefix: str) -> List[str]:
        try:
            response = await asyncio.to_thread(
                self.table.scan,
                FilterExpression='begins_with(config_id, :prefix)',
                ExpressionAttributeValues={':prefix': prefix}
            )
            return [item['config_id'] for item in response.get('Items', [])]
        except ClientError as e:
            logger.error(f"DynamoDB scan failed: {e}")
            return []

    async def delete(self, key: str) -> bool:
        try:
            await asyncio.to_thread(
                self.table.delete_item,
                Key={'config_id': key}
            )
            return True
        except ClientError as e:
            logger.error(f"DynamoDB delete failed: {e}")
            return False

    async def get_version(self, key: str) -> Optional[str]:
        item = await self.get(key)
        return item.get('version') if item else None


class S3ConfigStore(ConfigurationStore):
    """S3-based configuration store"""

    def __init__(self, bucket: str, region: str):
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client('s3', region_name=region)

    async def get(self, key: str) -> Optional[Dict]:
        try:
            response = await asyncio.to_thread(
                self.s3.get_object,
                Bucket=self.bucket,
                Key=key
            )
            content = await asyncio.to_thread(response['Body'].read)
            return json.loads(content)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            logger.error(f"S3 get failed: {e}")
            return None

    async def set(self, key: str, value: Dict, version: str) -> bool:
        try:
            content = json.dumps(value, indent=2)
            await asyncio.to_thread(
                self.s3.put_object,
                Bucket=self.bucket,
                Key=key,
                Body=content,
                Metadata={'version': version}
            )
            logger.info(f"Saved config {key} v{version} to S3")
            return True
        except ClientError as e:
            logger.error(f"S3 put failed: {e}")
            return False

    async def list(self, prefix: str) -> List[str]:
        try:
            response = await asyncio.to_thread(
                self.s3.list_objects_v2,
                Bucket=self.bucket,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logger.error(f"S3 list failed: {e}")
            return []

    async def delete(self, key: str) -> bool:
        try:
            await asyncio.to_thread(
                self.s3.delete_object,
                Bucket=self.bucket,
                Key=key
            )
            return True
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            return False

    async def get_version(self, key: str) -> Optional[str]:
        try:
            response = await asyncio.to_thread(
                self.s3.head_object,
                Bucket=self.bucket,
                Key=key
            )
            return response.get('Metadata', {}).get('version')
        except ClientError:
            return None


class ConfigSync:
    """
    Configuration Synchronization Service

    Manages synchronization of configuration data across multiple regions
    with version control and conflict detection.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.stores: Dict[str, ConfigurationStore] = {}
        self.sync_statuses: Dict[str, ConfigSyncStatus] = {}
        self.sync_interval = config.get('sync_interval_seconds', 60)
        self._running = False
        self._initialize_stores()

    def _initialize_stores(self):
        """Initialize configuration stores for each region"""

        for region_config in self.config.get('regions', []):
            region_name = region_config['name']
            store_type = region_config.get('config_store_type', 'dynamodb')

            if store_type == 'dynamodb':
                store = DynamoDBConfigStore(
                    table_name=region_config['config_table'],
                    region=region_config.get('aws_region', 'us-west-2')
                )
            elif store_type == 's3':
                store = S3ConfigStore(
                    bucket=region_config['config_bucket'],
                    region=region_config.get('aws_region', 'us-west-2')
                )
            else:
                raise ValueError(f"Unsupported store type: {store_type}")

            self.stores[region_name] = store
            logger.info(f"Initialized {store_type} config store for {region_name}")

    @staticmethod
    def compute_checksum(data: Dict) -> str:
        """Compute checksum of configuration data"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def create_config(
        self,
        name: str,
        config_type: ConfigType,
        data: Dict[str, Any],
        source_region: str,
        version: str = "1.0.0",
        encrypted: bool = False,
        tags: Optional[Dict[str, str]] = None
    ) -> ConfigItem:
        """Create a new configuration item"""

        config_id = f"{config_type.value}/{name}"
        checksum = self.compute_checksum(data)

        config_item = ConfigItem(
            config_id=config_id,
            name=name,
            config_type=config_type,
            data=data,
            version=version,
            checksum=checksum,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            source_region=source_region,
            encrypted=encrypted,
            tags=tags or {}
        )

        # Save to source region
        store = self.stores[source_region]
        success = await store.set(config_id, asdict(config_item), version)

        if success:
            logger.info(f"Created config {config_id} in {source_region}")
        else:
            raise Exception(f"Failed to create config {config_id}")

        return config_item

    async def update_config(
        self,
        config_id: str,
        data: Dict[str, Any],
        source_region: str,
        version: str
    ) -> ConfigItem:
        """Update an existing configuration"""

        store = self.stores[source_region]

        # Get existing config
        existing = await store.get(config_id)
        if not existing:
            raise ValueError(f"Config {config_id} not found")

        # Update fields
        checksum = self.compute_checksum(data)
        existing['data'] = data
        existing['version'] = version
        existing['checksum'] = checksum
        existing['updated_at'] = datetime.utcnow().isoformat()

        # Save
        success = await store.set(config_id, existing, version)

        if success:
            logger.info(f"Updated config {config_id} to v{version}")
            config_item = ConfigItem(**existing)
            return config_item
        else:
            raise Exception(f"Failed to update config {config_id}")

    async def get_config(
        self,
        config_id: str,
        region: str
    ) -> Optional[ConfigItem]:
        """Retrieve a configuration from a specific region"""

        store = self.stores[region]
        data = await store.get(config_id)

        if data:
            # Handle both dict and ConfigItem
            if isinstance(data.get('config_type'), str):
                data['config_type'] = ConfigType(data['config_type'])
            return ConfigItem(**data)

        return None

    async def sync_config(
        self,
        config_id: str,
        source_region: str,
        target_regions: List[str]
    ) -> Dict[str, ConfigSyncStatus]:
        """Synchronize a configuration to target regions"""

        results = {}
        source_store = self.stores[source_region]

        # Get config from source
        source_data = await source_store.get(config_id)
        if not source_data:
            raise ValueError(f"Config {config_id} not found in {source_region}")

        source_version = source_data.get('version')

        # Sync to each target region
        for target_region in target_regions:
            if target_region == source_region:
                continue

            status = ConfigSyncStatus(
                config_id=config_id,
                source_region=source_region,
                target_region=target_region,
                status="syncing",
                last_synced_at=None,
                version=source_version,
                error_message=None
            )

            try:
                target_store = self.stores[target_region]

                # Check if config exists in target
                target_data = await target_store.get(config_id)

                if target_data:
                    # Check for conflicts
                    target_version = target_data.get('version')
                    target_checksum = target_data.get('checksum')
                    source_checksum = source_data.get('checksum')

                    if target_version != source_version or target_checksum != source_checksum:
                        logger.warning(
                            f"Version/checksum mismatch for {config_id}: "
                            f"source={source_version}/{source_checksum}, "
                            f"target={target_version}/{target_checksum}"
                        )

                # Write config to target
                success = await target_store.set(
                    config_id,
                    source_data,
                    source_version
                )

                if success:
                    status.status = "synced"
                    status.last_synced_at = datetime.utcnow().isoformat()
                    logger.info(f"Synced {config_id} to {target_region}")
                else:
                    status.status = "error"
                    status.error_message = "Failed to write to target"

            except Exception as e:
                status.status = "error"
                status.error_message = str(e)
                logger.error(f"Failed to sync {config_id} to {target_region}: {e}")

            results[target_region] = status
            self.sync_statuses[f"{config_id}:{target_region}"] = status

        return results

    async def sync_all_configs(
        self,
        source_region: str,
        target_regions: List[str],
        config_type: Optional[ConfigType] = None
    ) -> Dict[str, Dict[str, ConfigSyncStatus]]:
        """Synchronize all configurations from source to target regions"""

        results = {}
        source_store = self.stores[source_region]

        # List all configs
        prefix = f"{config_type.value}/" if config_type else ""
        config_ids = await source_store.list(prefix)

        logger.info(f"Syncing {len(config_ids)} configs from {source_region}")

        # Sync each config
        for config_id in config_ids:
            try:
                sync_results = await self.sync_config(
                    config_id,
                    source_region,
                    target_regions
                )
                results[config_id] = sync_results
            except Exception as e:
                logger.error(f"Failed to sync {config_id}: {e}")

        return results

    async def verify_sync(
        self,
        config_id: str,
        regions: List[str]
    ) -> Dict[str, bool]:
        """Verify that a config is synchronized across regions"""

        results = {}
        checksums = {}

        # Get config and checksum from each region
        for region in regions:
            store = self.stores[region]
            data = await store.get(config_id)

            if data:
                checksums[region] = data.get('checksum')
            else:
                checksums[region] = None

        # Compare checksums
        reference_checksum = checksums.get(regions[0])

        for region in regions:
            checksum = checksums[region]
            in_sync = (checksum == reference_checksum) and checksum is not None
            results[region] = in_sync

            if not in_sync:
                logger.warning(
                    f"Config {config_id} out of sync in {region}: "
                    f"{checksum} vs {reference_checksum}"
                )

        return results

    async def continuous_sync(
        self,
        source_region: str,
        target_regions: List[str]
    ):
        """Continuously synchronize configurations"""

        self._running = True
        logger.info(
            f"Starting continuous config sync from {source_region} "
            f"to {target_regions} every {self.sync_interval}s"
        )

        while self._running:
            try:
                await self.sync_all_configs(source_region, target_regions)
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Error in continuous sync: {e}")
                await asyncio.sleep(self.sync_interval)

    def stop_continuous_sync(self):
        """Stop continuous synchronization"""
        self._running = False
        logger.info("Stopped continuous config sync")

    async def get_sync_status(
        self,
        config_id: str,
        target_region: Optional[str] = None
    ) -> Dict[str, ConfigSyncStatus]:
        """Get synchronization status for a configuration"""

        if target_region:
            key = f"{config_id}:{target_region}"
            status = self.sync_statuses.get(key)
            return {target_region: status} if status else {}

        # Return all statuses for this config
        results = {}
        for key, status in self.sync_statuses.items():
            if key.startswith(f"{config_id}:"):
                results[status.target_region] = status

        return results

    async def export_config(
        self,
        config_id: str,
        region: str,
        format: str = "json"
    ) -> str:
        """Export configuration to file format"""

        config = await self.get_config(config_id, region)

        if not config:
            raise ValueError(f"Config {config_id} not found in {region}")

        if format == "json":
            return json.dumps(asdict(config), indent=2)
        elif format == "yaml":
            return yaml.dump(asdict(config), default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def import_config(
        self,
        content: str,
        region: str,
        format: str = "json"
    ) -> ConfigItem:
        """Import configuration from file content"""

        if format == "json":
            data = json.loads(content)
        elif format == "yaml":
            data = yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Restore enum type
        if isinstance(data.get('config_type'), str):
            data['config_type'] = ConfigType(data['config_type'])

        config_item = ConfigItem(**data)

        # Save to region
        store = self.stores[region]
        await store.set(config_item.config_id, asdict(config_item), config_item.version)

        logger.info(f"Imported config {config_item.config_id} to {region}")
        return config_item
