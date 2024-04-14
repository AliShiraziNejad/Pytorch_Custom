import torch
from collections import defaultdict

"""
ASAM:
    rho-  0.05 to 1.0+ (start with 0.05 for AdamW)
    eta-  0.01
SAM:
    rho-  0.05 to 0.1 (start with 0.05 for AdamW)
"""

"""
Example configs:

ASAM(optimizer, model, rho=2.0, eta=0.01)
ASAM(optimizer, model, rho=1.0, eta=0.01)
ASAM(optimizer, model, rho=0.5, eta=0.01)
ASAM(optimizer, model, rho=0.1, eta=0.01)
ASAM(optimizer, model, rho=0.05, eta=0.01)

SAM(optimizer, model, rho=0.1)
SAM(optimizer, model, rho=0.05) <-- PREFERRED 
"""

"""
# Minimizer
optimizer = torch.optim.SGD(model.parameters())

minimizer = ASAM(optimizer, model, rho=rho, eta=eta)
or
minimizer = SAM(optimizer, model, rho=rho)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, epochs)

# Ascent Step
predictions = model(inputs)

batch_loss = criterion(predictions, targets)  # save as variable because this is the loss we measure our model on
batch_loss.backward()
minimizer.ascent_step()

# Descent Step
criterion(model(inputs), targets).backward()
minimizer.descent_step()
"""


class ASAM:
    """
    1) multiply the current gradients by [abs(current parameter) + eta] --> larger the parameter the bigger the scaling to the param's gradients
    2) calculate the new model's gradients overall L2 norm
    3) multiply the modified gradients by (rho / overall_grad_norm)
    4) add that result to the current parameters --> gradient ascent
    5) Calculate gradient with respect to the new (perturbed) position.
    6) Revert parameters to their original position and apply gradient descent using the gradients calculated at the perturbed position.
    """

    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        """
        Performs the adaptive perturbation step, adjusting parameters towards a 'sharper' position
        by scaling gradients based on their magnitude and a sharpness coefficient.
        """

        scaled_grads = []

        for param_name, param_tensor in self.model.named_parameters():  # looping through each layer in the network and getting the trainable parameters matrix
            if param_tensor.grad is None:
                continue

            scaled_params = self.state[param_tensor].get("epsilon")  # epsilon, the perturbation -- initializing it (gotta do it for every pass)

            if scaled_params is None:  # first forward pass
                scaled_params = torch.clone(param_tensor).detach()
                self.state[param_tensor]["epsilon"] = scaled_params  # placeholder tensor with the same shape as the scale params (allocating memory)

            if 'weight' in param_name:
                scaled_params[...] = param_tensor[...]
                scaled_params.abs_().add_(self.eta)  # get the magnitude of the parameters and add eta (to make the scaling non-zero)
                param_tensor.grad.mul_(scaled_params)  # scale the gradient based on how the magnitude of the parameters

            scaled_grads.append(torch.norm(param_tensor.grad, p=2))  # L2 Norm of the layer's scaled gradients (scalar)

        scaled_grads_norm = torch.norm(torch.stack(scaled_grads), p=2) + 1.e-16  # L2 Norm of the entire model's scaled gradients (scalar)

        for param_name, param_tensor in self.model.named_parameters():  # looping through each layer in the network and getting the trainable parameters matrix
            if param_tensor.grad is None:
                continue

            epsilon = self.state[param_tensor].get("epsilon")  # get the magnitude (plus eta) of the parameters

            if 'weight' in param_name:
                param_tensor.grad.mul_(epsilon)

            epsilon[...] = param_tensor.grad[...]  # make epsilon the scaled gradients for that layer
            epsilon.mul_(self.rho / scaled_grads_norm)  # Calculate the scaled perturbation for each parameter (worst-case)
            param_tensor.add_(epsilon)  # Apply the perturbation, moving each parameter to its 'worst-case' position Wadv = Wt + worst-case

        self.optimizer.zero_grad()  # Prepare for the next gradient computation by clearing previous gradients

    @torch.no_grad()
    def descent_step(self):
        """
        Performs the step back and updates parameters based on the gradient
        calculated at the 'worst-case' perturbed position.
        """
        for param_name, param_tensor in self.model.named_parameters():  # looping through each layer in the network and getting the trainable parameters matrix
            if param_tensor.grad is None:
                continue

            param_tensor.sub_(self.state[param_tensor]["epsilon"])  # get back to original Wt

        self.optimizer.step()  # update parameters based on the worst-case's gradient
        self.optimizer.zero_grad()  # Prepare for the next gradient computation by clearing previous gradients


class SAM(ASAM):
    """
    1) Compute gradient where you currently are
    2) multiply the current gradient by (rho / overall_grad_norm)
    3) add that result to the current parameters
    4) Calculate gradient with respect to the new (perturbed) position.
    5) Revert parameters to their original position and apply gradient descent using the gradients calculated at the perturbed position.
    """

    @torch.no_grad()
    def ascent_step(self):
        """
        Performs the perturbation step, moving parameters to a 'worst-case' position.
        """

        grads = []

        for param_name, param_tensor in self.model.named_parameters():  # looping through each layer in the network and getting the trainable parameters matrix
            if param_tensor.grad is None:
                continue

            grads.append(torch.linalg.norm(param_tensor.grad, ord=2))  # L2 Norm of the layer's gradients (scalar)

        grad_norm = torch.linalg.norm(torch.stack(grads), ord=2) + 1.e-16  # L2 Norm of the entire model's gradients (scalar)

        for param_name, param_tensor in self.model.named_parameters():  # looping through each layer in the network and getting the param matrix
            if param_tensor.grad is None:
                continue

            epsilon = self.state[param_tensor].get("epsilon")  # epsilon, the perturbation -- initializing it (gotta do it for every pass)

            # Retrieve or initialize the perturbation (eps) for each parameter
            if epsilon is None:  # first forward pass
                epsilon = torch.clone(param_tensor).detach()
                self.state[param_tensor]["epsilon"] = epsilon  # placeholder tensor with the same shape as the scale perturbation (allocating memory)

            epsilon[...] = param_tensor.grad[...]
            epsilon.mul_(self.rho / grad_norm)  # Calculate the scaled perturbation for each parameter (worst-case)
            param_tensor.add_(epsilon)  # Apply the perturbation, moving each parameter to its 'worst-case' position Wadv = Wt + worst-case

        self.optimizer.zero_grad()  # Prepare for the next gradient computation by clearing previous gradients

    # calculate gradients after this step at the new parameter positions
