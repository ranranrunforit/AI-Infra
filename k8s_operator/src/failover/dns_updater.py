"""
DNS Updater

Manages DNS updates for multi-region failover using Route53.
Supports multiple routing policies: failover, weighted, latency-based.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DNSRecord:
    """Represents a DNS record"""
    record_name: str
    record_type: str
    value: str
    ttl: int
    routing_policy: str  # simple, failover, weighted, latency, geolocation
    weight: Optional[int] = None
    set_identifier: Optional[str] = None
    health_check_id: Optional[str] = None


class DNSUpdater:
    """
    DNS Management for Multi-Region Failover

    Manages Route53 DNS records for traffic routing across regions.
    Supports automatic updates during failover events.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.route53 = boto3.client('route53', region_name='us-west-2')
        self.hosted_zone_id = config.get('hosted_zone_id')
        self.domain_name = config.get('domain_name')
        self.service_name = config.get('service_name', 'ml-platform')
        self.update_history: List[Dict] = []

    async def get_hosted_zone_id(self) -> str:
        """Get hosted zone ID from domain name"""
        if self.hosted_zone_id:
            return self.hosted_zone_id

        try:
            response = await asyncio.to_thread(
                self.route53.list_hosted_zones_by_name,
                DNSName=self.domain_name
            )

            for zone in response['HostedZones']:
                if zone['Name'].rstrip('.') == self.domain_name.rstrip('.'):
                    self.hosted_zone_id = zone['Id'].split('/')[-1]
                    logger.info(f"Found hosted zone: {self.hosted_zone_id}")
                    return self.hosted_zone_id

            raise ValueError(f"Hosted zone not found for {self.domain_name}")

        except ClientError as e:
            logger.error(f"Failed to get hosted zone: {e}")
            raise

    async def create_health_check(
        self,
        endpoint: str,
        region: str,
        port: int = 443,
        path: str = "/health"
    ) -> str:
        """Create a Route53 health check"""
        try:
            response = await asyncio.to_thread(
                self.route53.create_health_check,
                HealthCheckConfig={
                    'Type': 'HTTPS',
                    'ResourcePath': path,
                    'FullyQualifiedDomainName': endpoint,
                    'Port': port,
                    'RequestInterval': 30,
                    'FailureThreshold': 3,
                },
                CallerReference=f"{region}-{datetime.utcnow().timestamp()}"
            )

            health_check_id = response['HealthCheck']['Id']
            logger.info(f"Created health check {health_check_id} for {endpoint}")
            return health_check_id

        except ClientError as e:
            logger.error(f"Failed to create health check: {e}")
            raise

    async def update_health_check_status(
        self,
        health_check_id: str,
        enabled: bool
    ) -> bool:
        """Enable or disable a health check"""
        try:
            await asyncio.to_thread(
                self.route53.update_health_check,
                HealthCheckId=health_check_id,
                Disabled=not enabled
            )
            logger.info(f"Health check {health_check_id} {'enabled' if enabled else 'disabled'}")
            return True
        except ClientError as e:
            logger.error(f"Failed to update health check: {e}")
            return False

    async def update_primary_region(
        self,
        region: str,
        endpoint: str
    ) -> bool:
        """Update DNS to point to a new primary region"""
        try:
            zone_id = await self.get_hosted_zone_id()
            record_name = f"{self.service_name}.{self.domain_name}"

            # Create change batch
            changes = [{
                'Action': 'UPSERT',
                'ResourceRecordSet': {
                    'Name': record_name,
                    'Type': 'CNAME',
                    'TTL': 60,
                    'ResourceRecords': [{'Value': endpoint}],
                    'SetIdentifier': f'{region}-primary',
                    'Failover': 'PRIMARY'
                }
            }]

            response = await asyncio.to_thread(
                self.route53.change_resource_record_sets,
                HostedZoneId=zone_id,
                ChangeBatch={'Changes': changes}
            )

            change_id = response['ChangeInfo']['Id']

            # Wait for change to propagate
            await self._wait_for_change(change_id)

            self.update_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'update_primary',
                'region': region,
                'endpoint': endpoint,
                'change_id': change_id
            })

            logger.info(f"Updated primary DNS to {region}: {endpoint}")
            return True

        except ClientError as e:
            logger.error(f"Failed to update primary region: {e}")
            return False

    async def update_weighted_routing(
        self,
        region: str,
        weight: int
    ) -> bool:
        """Update weighted routing for a region"""
        try:
            zone_id = await self.get_hosted_zone_id()
            record_name = f"weighted.{self.service_name}.{self.domain_name}"

            region_config = next(
                r for r in self.config.get('regions', [])
                if r['name'] == region
            )
            endpoint = region_config['endpoint']

            changes = [{
                'Action': 'UPSERT',
                'ResourceRecordSet': {
                    'Name': record_name,
                    'Type': 'CNAME',
                    'TTL': 60,
                    'ResourceRecords': [{'Value': endpoint}],
                    'SetIdentifier': region,
                    'Weight': weight
                }
            }]

            response = await asyncio.to_thread(
                self.route53.change_resource_record_sets,
                HostedZoneId=zone_id,
                ChangeBatch={'Changes': changes}
            )

            await self._wait_for_change(response['ChangeInfo']['Id'])

            logger.info(f"Updated weight for {region} to {weight}")
            return True

        except ClientError as e:
            logger.error(f"Failed to update weighted routing: {e}")
            return False

    async def update_failover_config(
        self,
        primary_region: str,
        secondary_region: str
    ) -> bool:
        """Update failover configuration"""
        try:
            zone_id = await self.get_hosted_zone_id()
            record_name = f"{self.service_name}.{self.domain_name}"

            # Get region configs
            primary_config = next(
                r for r in self.config['regions']
                if r['name'] == primary_region
            )
            secondary_config = next(
                r for r in self.config['regions']
                if r['name'] == secondary_region
            )

            # Create health checks if needed
            primary_health_check = await self.create_health_check(
                endpoint=primary_config['endpoint'],
                region=primary_region
            )

            secondary_health_check = await self.create_health_check(
                endpoint=secondary_config['endpoint'],
                region=secondary_region
            )

            # Create change batch for both records
            changes = [
                {
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': record_name,
                        'Type': 'CNAME',
                        'TTL': 60,
                        'ResourceRecords': [{'Value': primary_config['endpoint']}],
                        'SetIdentifier': f'{primary_region}-primary',
                        'Failover': 'PRIMARY',
                        'HealthCheckId': primary_health_check
                    }
                },
                {
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': record_name,
                        'Type': 'CNAME',
                        'TTL': 60,
                        'ResourceRecords': [{'Value': secondary_config['endpoint']}],
                        'SetIdentifier': f'{secondary_region}-secondary',
                        'Failover': 'SECONDARY',
                        'HealthCheckId': secondary_health_check
                    }
                }
            ]

            response = await asyncio.to_thread(
                self.route53.change_resource_record_sets,
                HostedZoneId=zone_id,
                ChangeBatch={'Changes': changes}
            )

            await self._wait_for_change(response['ChangeInfo']['Id'])

            logger.info(
                f"Updated failover config: primary={primary_region}, "
                f"secondary={secondary_region}"
            )
            return True

        except ClientError as e:
            logger.error(f"Failed to update failover config: {e}")
            return False

    async def update_latency_routing(
        self,
        region_endpoints: Dict[str, str]
    ) -> bool:
        """Update latency-based routing for all regions"""
        try:
            zone_id = await self.get_hosted_zone_id()
            record_name = f"lb.{self.service_name}.{self.domain_name}"

            changes = []

            for region, endpoint in region_endpoints.items():
                # Map region name to AWS region
                region_config = next(
                    r for r in self.config['regions']
                    if r['name'] == region
                )
                aws_region = region_config.get('aws_region', 'us-west-2')

                changes.append({
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': record_name,
                        'Type': 'CNAME',
                        'TTL': 60,
                        'ResourceRecords': [{'Value': endpoint}],
                        'SetIdentifier': region,
                        'Region': aws_region
                    }
                })

            response = await asyncio.to_thread(
                self.route53.change_resource_record_sets,
                HostedZoneId=zone_id,
                ChangeBatch={'Changes': changes}
            )

            await self._wait_for_change(response['ChangeInfo']['Id'])

            logger.info(f"Updated latency routing for {len(region_endpoints)} regions")
            return True

        except ClientError as e:
            logger.error(f"Failed to update latency routing: {e}")
            return False

    async def remove_region_from_dns(self, region: str) -> bool:
        """Remove a region from DNS rotation"""
        try:
            zone_id = await self.get_hosted_zone_id()

            # List all record sets
            response = await asyncio.to_thread(
                self.route53.list_resource_record_sets,
                HostedZoneId=zone_id
            )

            # Find records with this region's set identifier
            changes = []
            for record in response['ResourceRecordSets']:
                if record.get('SetIdentifier', '').startswith(region):
                    changes.append({
                        'Action': 'DELETE',
                        'ResourceRecordSet': record
                    })

            if changes:
                await asyncio.to_thread(
                    self.route53.change_resource_record_sets,
                    HostedZoneId=zone_id,
                    ChangeBatch={'Changes': changes}
                )
                logger.info(f"Removed {region} from DNS rotation")

            return True

        except ClientError as e:
            logger.error(f"Failed to remove region from DNS: {e}")
            return False

    async def _wait_for_change(
        self,
        change_id: str,
        timeout_seconds: int = 300
    ):
        """Wait for a DNS change to propagate"""
        start_time = datetime.utcnow()

        while True:
            try:
                response = await asyncio.to_thread(
                    self.route53.get_change,
                    Id=change_id
                )

                status = response['ChangeInfo']['Status']

                if status == 'INSYNC':
                    logger.info(f"DNS change {change_id} propagated")
                    return

                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    logger.warning(f"DNS change {change_id} timed out after {elapsed}s")
                    return

                await asyncio.sleep(5)

            except ClientError as e:
                logger.error(f"Error checking change status: {e}")
                return

    async def get_current_dns_config(self) -> List[Dict]:
        """Get current DNS configuration"""
        try:
            zone_id = await self.get_hosted_zone_id()

            response = await asyncio.to_thread(
                self.route53.list_resource_record_sets,
                HostedZoneId=zone_id
            )

            records = []
            for record in response['ResourceRecordSets']:
                if self.service_name in record['Name']:
                    records.append({
                        'name': record['Name'],
                        'type': record['Type'],
                        'ttl': record.get('TTL'),
                        'values': record.get('ResourceRecords', []),
                        'set_identifier': record.get('SetIdentifier'),
                        'routing_policy': self._get_routing_policy(record)
                    })

            return records

        except ClientError as e:
            logger.error(f"Failed to get DNS config: {e}")
            return []

    @staticmethod
    def _get_routing_policy(record: Dict) -> str:
        """Determine routing policy from record"""
        if 'Weight' in record:
            return 'weighted'
        elif 'Failover' in record:
            return 'failover'
        elif 'Region' in record:
            return 'latency'
        elif 'GeoLocation' in record:
            return 'geolocation'
        else:
            return 'simple'

    def get_update_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get DNS update history"""
        if limit:
            return self.update_history[-limit:]
        return self.update_history

    async def test_dns_resolution(self, record_name: str) -> Dict:
        """Test DNS resolution for a record"""
        import socket

        try:
            full_name = f"{record_name}.{self.domain_name}"
            ips = await asyncio.to_thread(socket.gethostbyname_ex, full_name)

            return {
                'record': full_name,
                'resolved': True,
                'ips': ips[2],
                'aliases': ips[1]
            }

        except socket.gaierror as e:
            return {
                'record': f"{record_name}.{self.domain_name}",
                'resolved': False,
                'error': str(e)
            }
