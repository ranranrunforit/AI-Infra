/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TrainingJobSpec defines the desired state of TrainingJob
type TrainingJobSpec struct {
	// Model architecture to train (e.g., resnet50, bert-base, gpt2)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Model string `json:"model"`

	// Dataset to use for training
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Dataset string `json:"dataset"`

	// Number of worker replicas for distributed training
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100
	NumWorkers int32 `json:"numWorkers"`

	// Number of GPUs per worker
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=16
	// +kubebuilder:default=1
	// +optional
	GPUsPerWorker int32 `json:"gpusPerWorker,omitempty"`

	// ML framework (pytorch, tensorflow, jax)
	// +kubebuilder:validation:Enum=pytorch;tensorflow;jax
	// +kubebuilder:default=pytorch
	// +optional
	Framework string `json:"framework,omitempty"`

	// Container image for training
	// +kubebuilder:default="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
	// +optional
	Image string `json:"image,omitempty"`

	// Command to run in the container
	// +optional
	Command []string `json:"command,omitempty"`

	// Arguments for the command
	// +optional
	Args []string `json:"args,omitempty"`

	// Environment variables
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// Resource requirements
	// +optional
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`

	// Training hyperparameters
	// +optional
	Hyperparameters map[string]string `json:"hyperparameters,omitempty"`

	// Checkpointing configuration
	// +optional
	Checkpoint *CheckpointConfig `json:"checkpoint,omitempty"`

	// Scheduling configuration
	// +optional
	Scheduling *SchedulingConfig `json:"scheduling,omitempty"`

	// Monitoring configuration
	// +optional
	Monitoring *MonitoringConfig `json:"monitoring,omitempty"`

	// Networking configuration for distributed training
	// +optional
	Networking *NetworkingConfig `json:"networking,omitempty"`
}

type CheckpointConfig struct {
	// Enable checkpointing
	// +kubebuilder:default=true
	Enabled bool `json:"enabled"`

	// Checkpoint every N epochs
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=5
	Frequency int32 `json:"frequency,omitempty"`

	// Storage backend for checkpoints
	// +optional
	Storage *StorageConfig `json:"storage,omitempty"`

	// Path to checkpoint to resume from
	// +optional
	ResumeFrom string `json:"resumeFrom,omitempty"`
}

type StorageConfig struct {
	// +kubebuilder:validation:Enum=pvc;s3;gcs;nfs
	// +kubebuilder:default=pvc
	Type string `json:"type"`

	// +optional
	PVCName string `json:"pvcName,omitempty"`
	// +optional
	S3Bucket string `json:"s3Bucket,omitempty"`
	// +optional
	GCSBucket string `json:"gcsBucket,omitempty"`
	// +optional
	NFSServer string `json:"nfsServer,omitempty"`
	// +optional
	NFSPath string `json:"nfsPath,omitempty"`
}

type SchedulingConfig struct {
	// Priority class name
	// +optional
	Priority string `json:"priority,omitempty"`

	// Node selector labels
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Tolerations for taints
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`
}

type MonitoringConfig struct {
	// +kubebuilder:default=true
	Enabled bool `json:"enabled"`

	// +kubebuilder:default=8080
	// +kubebuilder:validation:Minimum=1024
	MetricsPort int32 `json:"metricsPort,omitempty"`

	// +optional
	MLFlowTrackingURI string `json:"mlflowTrackingUri,omitempty"`
	// +optional
	WandBProject string `json:"wandbProject,omitempty"`
}

type NetworkingConfig struct {
	// +kubebuilder:validation:Enum=nccl;gloo;mpi
	// +kubebuilder:default=nccl
	Backend string `json:"backend"`

	// +kubebuilder:default=29500
	MasterPort int32 `json:"masterPort"`
}

// TrainingJobStatus defines the observed state of TrainingJob
type TrainingJobStatus struct {
	// Current state of the training job
	// +kubebuilder:validation:Enum=Pending;Initializing;Running;Completed;Failed;Suspended
	State string `json:"state,omitempty"`

	// Current conditions of the training job
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// Training progress percentage
	Progress string `json:"progress,omitempty"`

	// Current training epoch
	CurrentEpoch int32 `json:"currentEpoch,omitempty"`

	// Allocated resources
	Resources ResourceStatus `json:"resources,omitempty"`

	// Status of individual workers
	Workers WorkerStatus `json:"workers,omitempty"`

	// Time when training started
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// Time when training completed
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`
}

type ResourceStatus struct {
	AllocatedGPUs  int32 `json:"allocatedGPUs,omitempty"`
	AllocatedNodes int32 `json:"allocatedNodes,omitempty"`
}

type WorkerStatus struct {
	Active    int32 `json:"active"`
	Succeeded int32 `json:"succeeded"`
	Failed    int32 `json:"failed"`
	Pending   int32 `json:"pending"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="State",type="string",JSONPath=".status.state"
// +kubebuilder:printcolumn:name="Progress",type="string",JSONPath=".status.progress"
// +kubebuilder:printcolumn:name="Epoch",type="integer",JSONPath=".status.currentEpoch"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// TrainingJob is the Schema for the trainingjobs API
type TrainingJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TrainingJobSpec   `json:"spec,omitempty"`
	Status TrainingJobStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TrainingJobList contains a list of TrainingJob
type TrainingJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TrainingJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TrainingJob{}, &TrainingJobList{})
}
