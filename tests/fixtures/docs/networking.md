# Kubernetes Networking

Networking in Kubernetes is built on services and ingress controllers.
Pods get individual IPs but services give them stable endpoints.

## Service Types

A ClusterIP service is reachable only inside the cluster.
A NodePort service exposes the service on each node's IP at a static port.
A LoadBalancer service provisions an external load balancer.
