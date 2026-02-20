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

package controller

import (
	"context"
	"fmt"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	mlv1 "github.com/example/k8s-operator/api/v1"
	"github.com/example/k8s-operator/internal/metrics"
	"github.com/example/k8s-operator/internal/resources"
)

// TrainingJobReconciler reconciles a TrainingJob object
type TrainingJobReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=ml.example.com,resources=trainingjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=ml.example.com,resources=trainingjobs/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=ml.example.com,resources=trainingjobs/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services;configmaps;events;pods,verbs=get;list;watch;create;update;patch;delete

func (r *TrainingJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	// Fetch the TrainingJob instance
	trainingJob := &mlv1.TrainingJob{}
	err := r.Get(ctx, req.NamespacedName, trainingJob)
	if err != nil {
		if errors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			return ctrl.Result{}, nil
		}
		// Error reading the object - requeue the request.
		return ctrl.Result{}, err
	}

	// Initialize status if needed
	if trainingJob.Status.State == "" {
		trainingJob.Status.State = "Pending"
		if err := r.Status().Update(ctx, trainingJob); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Check if Job already exists, if not create a new one
	foundJob := &batchv1.Job{}
	err = r.Get(ctx, client.ObjectKey{Name: trainingJob.Name + "-training", Namespace: trainingJob.Namespace}, foundJob)
	if err != nil && errors.IsNotFound(err) {
		// Define a new Job
		jobBuilder := resources.NewJobBuilder()
		job := jobBuilder.BuildJob(trainingJob)
		
		// Set TrainingJob instance as the owner and controller
		if err := ctrl.SetControllerReference(trainingJob, job, r.Scheme); err != nil {
			return ctrl.Result{}, err
		}

		log.Info("Creating a new Job", "Job.Namespace", job.Namespace, "Job.Name", job.Name)
		if err := r.Create(ctx, job); err != nil {
			return ctrl.Result{}, err
		}

		// Update metrics
		metrics.JobCreated.WithLabelValues(trainingJob.Namespace).Inc()
		metrics.AllocatedGPUs.WithLabelValues(trainingJob.Namespace, trainingJob.Name).Set(float64(trainingJob.Spec.NumWorkers * int32(trainingJob.Spec.GPUsPerWorker)))

		// Create ConfigMap
		cmBuilder := resources.NewConfigMapBuilder()
		cm := cmBuilder.BuildConfigMap(trainingJob)
		if err := ctrl.SetControllerReference(trainingJob, cm, r.Scheme); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.Create(ctx, cm); err != nil {
			return ctrl.Result{}, err
		}

		// Create Service if needed
		if trainingJob.Spec.NumWorkers > 1 {
			svcBuilder := resources.NewServiceBuilder()
			svc := svcBuilder.BuildService(trainingJob)
			if err := ctrl.SetControllerReference(trainingJob, svc, r.Scheme); err != nil {
				return ctrl.Result{}, err
			}
			if err := r.Create(ctx, svc); err != nil {
				return ctrl.Result{}, err
			}
		}

		r.Recorder.Event(trainingJob, corev1.EventTypeNormal, "JobCreated", "Created Training Job")
		
		// Update status to Initializing
		trainingJob.Status.State = "Initializing"
		if err := r.Status().Update(ctx, trainingJob); err != nil {
			return ctrl.Result{}, err
		}

		return ctrl.Result{Requeue: true}, nil
	} else if err != nil {
		return ctrl.Result{}, err
	}

	// Update status based on Job status
	if trainingJob.Status.State != "Completed" && trainingJob.Status.State != "Failed" {
		if foundJob.Status.Succeeded > 0 {
			// Job completed
			if trainingJob.Status.State != "Completed" {
				trainingJob.Status.State = "Completed"
				trainingJob.Status.CompletionTime = foundJob.Status.CompletionTime
				r.Recorder.Event(trainingJob, corev1.EventTypeNormal, "JobCompleted", "Training Job Completed Successfully")
			}
		} else if foundJob.Status.Failed > 0 {
			// Check failure policy
			backoffLimit := int32(3)
			if trainingJob.Spec.FailurePolicy != nil && trainingJob.Spec.FailurePolicy.BackoffLimit != 0 {
				backoffLimit = trainingJob.Spec.FailurePolicy.BackoffLimit
			}
			
			if foundJob.Status.Failed >= backoffLimit {
				trainingJob.Status.State = "Failed"
				trainingJob.Status.FailureReason = "TooManyFailures"
				trainingJob.Status.FailureMessage = fmt.Sprintf("%d workers failed (backoff limit: %d)", foundJob.Status.Failed, backoffLimit)
				r.Recorder.Event(trainingJob, corev1.EventTypeWarning, "JobFailed", "Training Job Failed")
				metrics.JobFailed.WithLabelValues(trainingJob.Namespace, "TooManyFailures").Inc()
			}
		} else if foundJob.Status.Active > 0 {
			if trainingJob.Status.State != "Running" {
				trainingJob.Status.State = "Running"
				trainingJob.Status.StartTime = foundJob.Status.StartTime
				r.Recorder.Event(trainingJob, corev1.EventTypeNormal, "JobRunning", "Training Job is Running")
			}
		}

		// Update worker status
		trainingJob.Status.Workers.Active = foundJob.Status.Active
		trainingJob.Status.Workers.Succeeded = foundJob.Status.Succeeded
		trainingJob.Status.Workers.Failed = foundJob.Status.Failed
		// Calculate pending
		// This is a simplification; a more accurate pending count would require listing pods
		
		// Update metrics
		metrics.ActiveWorkers.WithLabelValues(trainingJob.Namespace, trainingJob.Name).Set(float64(foundJob.Status.Active))

		if err := r.Status().Update(ctx, trainingJob); err != nil {
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *TrainingJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&mlv1.TrainingJob{}).
		Owns(&batchv1.Job{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.ConfigMap{}).
		Complete(r)
}
