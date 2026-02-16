package resources

import (
	"encoding/json"
	"fmt"
	"strconv"

	mlv1 "github.com/example/k8s-operator/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ConfigMapBuilder builds Kubernetes ConfigMap resources
type ConfigMapBuilder struct{}

func NewConfigMapBuilder() *ConfigMapBuilder {
	return &ConfigMapBuilder{}
}

func (b *ConfigMapBuilder) BuildConfigMap(instance *mlv1.TrainingJob) *corev1.ConfigMap {
	ls := labelsForTrainingJob(instance.Name)
	configMapName := fmt.Sprintf("%s-config", instance.Name)

	data := map[string]string{
		"model":           instance.Spec.Model,
		"dataset":         instance.Spec.Dataset,
		"num_workers":     strconv.Itoa(int(instance.Spec.NumWorkers)),
		"gpus_per_worker": strconv.Itoa(int(instance.Spec.GPUsPerWorker)),
		"framework":       instance.Spec.Framework,
	}

	if instance.Spec.Hyperparameters != nil {
		hyperBytes, err := json.Marshal(instance.Spec.Hyperparameters)
		if err == nil {
			data["hyperparameters"] = string(hyperBytes)
		}
	}

	if instance.Spec.Networking != nil {
		data["backend"] = instance.Spec.Networking.Backend
		data["master_port"] = strconv.Itoa(int(instance.Spec.Networking.MasterPort))
	}

	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: instance.Namespace,
			Labels:    ls,
		},
		Data: data,
	}
}
