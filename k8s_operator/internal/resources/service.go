package resources

import (
	mlv1 "github.com/example/k8s-operator/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// ServiceBuilder builds Kubernetes Service resources
type ServiceBuilder struct{}

func NewServiceBuilder() *ServiceBuilder {
	return &ServiceBuilder{}
}

func (b *ServiceBuilder) BuildService(instance *mlv1.TrainingJob) *corev1.Service {
	ls := labelsForTrainingJob(instance.Name)
	serviceName := instance.Name + "-headless"

	masterPort := int32(29500)
	if instance.Spec.Networking != nil {
		masterPort = instance.Spec.Networking.MasterPort
	}

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: instance.Namespace,
			Labels:    ls,
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: "None", // Headless service
			Selector:  ls,
			Ports: []corev1.ServicePort{
				{
					Name:       "master",
					Port:       masterPort,
					TargetPort: intstr.FromInt(int(masterPort)),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}
}
