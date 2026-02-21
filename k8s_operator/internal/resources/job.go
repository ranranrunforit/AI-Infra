package resources

import (
	"fmt"
	"strconv"

	mlv1 "github.com/example/k8s-operator/api/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// JobBuilder builds Kubernetes Job resources
type JobBuilder struct{}

func NewJobBuilder() *JobBuilder {
	return &JobBuilder{}
}

func (b *JobBuilder) BuildJob(instance *mlv1.TrainingJob) *batchv1.Job {
	ls := labelsForTrainingJob(instance.Name)
	jobName := fmt.Sprintf("%s-training", instance.Name)

	podTemplate := b.buildPodTemplate(instance, ls)
	
	// Default backoff limit
	backoffLimit := int32(3)

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: instance.Namespace,
			Labels:    ls,
		},
		Spec: batchv1.JobSpec{
			Parallelism:  &instance.Spec.NumWorkers,
			Completions:  &instance.Spec.NumWorkers,
			BackoffLimit: &backoffLimit,
			Template:     *podTemplate,
			// For indexed jobs (needed for torch distributed sometimes), but we use simple parallelism here
			// as the Python original did. 
			// Check Python original: parallelism=num_workers, completions=num_workers
		},
	}

	return job
}

func (b *JobBuilder) buildPodTemplate(instance *mlv1.TrainingJob, labels map[string]string) *corev1.PodTemplateSpec {
	container := b.buildContainer(instance)
	volumes := b.buildVolumes(instance)

	podSpec := corev1.PodSpec{
		Containers:    []corev1.Container{*container},
		RestartPolicy: corev1.RestartPolicyOnFailure,
		Volumes:       volumes,
	}

	if instance.Spec.Scheduling != nil {
		podSpec.NodeSelector = instance.Spec.Scheduling.NodeSelector
		podSpec.Tolerations = instance.Spec.Scheduling.Tolerations
		if instance.Spec.Scheduling.Priority != "" {
			podSpec.PriorityClassName = instance.Spec.Scheduling.Priority
		}
	}

	return &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: labels,
		},
		Spec: podSpec,
	}
}

func (b *JobBuilder) buildContainer(instance *mlv1.TrainingJob) *corev1.Container {
	image := instance.Spec.Image
	if image == "" {
		image = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
	}

	command := instance.Spec.Command
	args := instance.Spec.Args

	// Default command for PyTorch if not specified
	if len(command) == 0 && (instance.Spec.Framework == "pytorch" || instance.Spec.Framework == "") {
		command = []string{"python", "-m", "torch.distributed.run"}
		if len(args) == 0 {
			masterPort := int32(29500)
			if instance.Spec.Networking != nil {
				masterPort = instance.Spec.Networking.MasterPort
			}
			args = []string{
				fmt.Sprintf("--nproc_per_node=%d", instance.Spec.GPUsPerWorker),
				"--nnodes=$(NUM_WORKERS)",
				"--node_rank=$(WORKER_RANK)",
				"--master_addr=$(MASTER_ADDR)",
				fmt.Sprintf("--master_port=%d", masterPort),
				"train.py",
			}
		}
	}

	envVars := b.buildEnvVars(instance)
	volumeMounts := b.buildVolumeMounts(instance)

	return &corev1.Container{
		Name:         "training",
		Image:        image,
		Command:      command,
		Args:         args,
		Env:          envVars,
		Resources:    instance.Spec.Resources,
		VolumeMounts: volumeMounts,
	}
}

func (b *JobBuilder) buildEnvVars(instance *mlv1.TrainingJob) []corev1.EnvVar {
	envVars := []corev1.EnvVar{
		{
			Name:  "NUM_WORKERS",
			Value: strconv.Itoa(int(instance.Spec.NumWorkers)),
		},
		{
			Name:  "GPUS_PER_WORKER",
			Value: strconv.Itoa(int(instance.Spec.GPUsPerWorker)),
		},
		{
			Name:  "MASTER_ADDR",
			Value: fmt.Sprintf("%s-headless", instance.Name),
		},
		{
			Name: "POD_NAME",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.name",
				},
			},
		},
		{
			Name: "POD_NAMESPACE",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.namespace",
				},
			},
		},
	}

	// User defined env vars
	envVars = append(envVars, instance.Spec.Env...)

	// Monitoring
	if instance.Spec.Monitoring != nil && instance.Spec.Monitoring.Enabled {
		if instance.Spec.Monitoring.MLFlowTrackingURI != "" {
			envVars = append(envVars, corev1.EnvVar{
				Name:  "MLFLOW_TRACKING_URI",
				Value: instance.Spec.Monitoring.MLFlowTrackingURI,
			})
		}
		if instance.Spec.Monitoring.WandBProject != "" {
			envVars = append(envVars, corev1.EnvVar{
				Name:  "WANDB_PROJECT",
				Value: instance.Spec.Monitoring.WandBProject,
			})
		}
	}

	return envVars
}

func (b *JobBuilder) buildVolumes(instance *mlv1.TrainingJob) []corev1.Volume {
	volumes := []corev1.Volume{
		{
			Name: "config",
			VolumeSource: corev1.VolumeSource{
				ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: fmt.Sprintf("%s-config", instance.Name),
					},
				},
			},
		},
	}

	if instance.Spec.Checkpoint != nil && instance.Spec.Checkpoint.Enabled {
		if instance.Spec.Checkpoint.Storage != nil && instance.Spec.Checkpoint.Storage.Type == "pvc" {
			pvcName := instance.Spec.Checkpoint.Storage.PVCName
			if pvcName == "" {
				pvcName = fmt.Sprintf("%s-checkpoints", instance.Name)
			}
			volumes = append(volumes, corev1.Volume{
				Name: "checkpoints",
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: pvcName,
					},
				},
			})
		}
		// Implement NFS, S3 etc. logic here similar to python
	}

	return volumes
}

func (b *JobBuilder) buildVolumeMounts(instance *mlv1.TrainingJob) []corev1.VolumeMount {
	mounts := []corev1.VolumeMount{
		{
			Name:      "config",
			MountPath: "/etc/training-config",
			ReadOnly:  true,
		},
	}

	// Only mount checkpoints volume if storage is explicitly configured as pvc
	// (must match the condition in buildVolumes)
	if instance.Spec.Checkpoint != nil && instance.Spec.Checkpoint.Enabled &&
		instance.Spec.Checkpoint.Storage != nil && instance.Spec.Checkpoint.Storage.Type == "pvc" {
		mounts = append(mounts, corev1.VolumeMount{
			Name:      "checkpoints",
			MountPath: "/checkpoints",
		})
	}

	return mounts
}

func labelsForTrainingJob(name string) map[string]string {
	return map[string]string{
		"app":          "training-job",
		"training-job": name,
		"component":    "worker",
	}
}
