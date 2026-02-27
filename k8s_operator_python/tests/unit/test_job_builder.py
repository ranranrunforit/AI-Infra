import unittest
from unittest.mock import MagicMock, patch
from kubernetes import client
from src.resources.job_builder import JobBuilder

class TestJobBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = JobBuilder()
        self.spec = {
            'model': 'resnet50',
            'dataset': 'imagenet',
            'numWorkers': 2,
            'gpusPerWorker': 1,
            'image': 'test-image:latest',
            'scheduling': {
                'nodeSelector': {'accelerator': 'nvidia-v100'},
                'tolerations': [
                    {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'}
                ]
            }
        }

    def test_build_job_structure(self):
        job = self.builder.build_job('test-job', 'default', self.spec)
        self.assertIsInstance(job, client.V1Job)
        self.assertEqual(job.metadata.name, 'test-job-training')
        self.assertEqual(job.spec.parallelism, 2)
        self.assertEqual(job.spec.completions, 2)

    def test_build_pod_template(self):
        template = self.builder._build_pod_template('test-job', 'default', self.spec)
        self.assertIsInstance(template, client.V1PodTemplateSpec)
        self.assertEqual(template.spec.containers[0].image, 'test-image:latest')

    def test_node_selector(self):
        # Mocking ApiClient to avoid needing a cluster config or connection
        with patch('kubernetes.client.ApiClient') as MockApiClient:
            mock_client = MockApiClient.return_value
            mock_client._ApiClient__deserialize.return_value = client.V1NodeSelector(
                node_selector_terms=[client.V1NodeSelectorTerm(
                    match_expressions=[client.V1NodeSelectorRequirement(
                        key='accelerator', operator='In', values=['nvidia-v100']
                    )]
                )]
            )
            
            selector = self.builder._build_node_selector(self.spec['scheduling']['nodeSelector'])
            self.assertIsNotNone(selector)

    def test_tolerations(self):
        with patch('kubernetes.client.ApiClient') as MockApiClient:
            mock_client = MockApiClient.return_value
            mock_client._ApiClient__deserialize.side_effect = [
                client.V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule')
            ]

            tols = self.builder._build_tolerations(self.spec)
            self.assertEqual(len(tols), 1)
            self.assertEqual(tols[0].key, 'nvidia.com/gpu')

    def test_resources(self):
        res = self.builder._build_resources(self.spec)
        self.assertEqual(res.limits['nvidia.com/gpu'], '1')
        self.assertEqual(res.requests['memory'], '16Gi')

if __name__ == '__main__':
    unittest.main()
