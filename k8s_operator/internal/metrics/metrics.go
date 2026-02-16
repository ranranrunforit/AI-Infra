package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

var (
	JobCreated = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "trainingjob_created_total",
			Help: "Total number of training jobs created",
		},
		[]string{"namespace"},
	)

	JobFailed = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "trainingjob_failed_total",
			Help: "Total number of failed training jobs",
		},
		[]string{"namespace", "reason"},
	)

	ActiveWorkers = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "trainingjob_active_workers",
			Help: "Number of active workers per training job",
		},
		[]string{"namespace", "training_job"},
	)

	AllocatedGPUs = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "trainingjob_allocated_gpus",
			Help: "Total number of GPUs allocated per training job",
		},
		[]string{"namespace", "training_job"},
	)
)

func init() {
	// Register custom metrics with the global prometheus registry
	metrics.Registry.MustRegister(JobCreated, JobFailed, ActiveWorkers, AllocatedGPUs)
}
