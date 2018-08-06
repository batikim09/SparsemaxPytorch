from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Sparsemax(nn.Module):
    def __init__(self, num_clusters, num_neurons_per_cluster):
        super(Sparsemax, self).__init__()
        self.num_clusters = num_clusters
        self.num_neurons_per_cluster = num_neurons_per_cluster
        
    def forward(self, input):

        input_reshape = torch.zeros(input.size())
        input_reshape = input.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        dim = 2
        #translate for numerical stability
        input_shift = input_reshape # - torch.max(input_reshape, dim)[0].expand_as(input_reshape)
        #print("DEBUG: input_reshape: ", input_reshape.size())

        #sorting input in descending order --> z(k)
        z_sorted = torch.sort(input_shift, dim=dim, descending=True)[0]
        #print("DEBUG: z_sorted: ", z_sorted.size())
        #number of classes (K) 
        input_size = input_shift.size()[dim]	        

        #range values (k = 1... K)
        range_values = Variable(torch.arange(1, input_size+1), requires_grad=False).cuda()
        range_values = range_values.expand_as(z_sorted).float()
        #print("DEBUG: range_values: ", range_values.size())

        #Determine sparsity of projection
        #b(0,...,0)
        bound = Variable(torch.zeros(z_sorted.size()),requires_grad=False).cuda()

        #A) 1 + b + k.z(k)
        bound = 1 + torch.addcmul(bound, range_values, z_sorted)        

        #B) cummulative sum_z(k): sum_z(2) = z(0) + z(1) + z(2)
        cumlative_sum_zs = torch.cumsum(z_sorted, dim)
        #print("DEBUG: cumlative_sum_zs: ", cumlative_sum_zs.size())

        # A > B 
        is_gt = torch.gt(bound, cumlative_sum_zs).type(torch.FloatTensor).cuda()
        #print("DEBUG: is_gt: ", is_gt.size())

        valid = Variable(torch.zeros(range_values.size()),requires_grad=False).cuda()
        
        # if A(k) > B(k) return k otherwise 0
        valid = torch.addcmul(valid, range_values, is_gt)

        #find max among K --> k_max
        k_max = torch.max(valid, dim)[0]
        #print("DEBUG: k_max: ", k_max.size())

        zs_sparse = Variable(torch.zeros(z_sorted.size()),requires_grad=False).cuda()
        
        # if A(k) > B(k) return z(k) otherwise 0 
        zs_sparse = torch.addcmul(zs_sparse, is_gt, z_sorted)

        # sum of zs_sparse - 1, z_sparse is already sorted; it is sum of z(k), k= 0~k_max
        # -1 prevents that too many z becomes zeros
        sum_zs = (torch.sum(zs_sparse, dim) - 1)

        taus = Variable(torch.zeros(k_max.size()),requires_grad=False).cuda()

        #tau(z) = (sum_zs_sparse - 1) / k_max
        #tau(z) is a shifted mean of z(k), upto k_max
        taus = torch.addcdiv(taus, sum_zs, k_max)
        #print("DEBUG: taus: ", taus.size())

        #expand is to copy matrix and paste number of times, copied matrixes are appended at the front!
        #We can't extend 16x1 --> 16x1x10, so we need a following trick: 
        taus_expanded = taus.expand_as(input_reshape.permute(2,0,1))
        taus_expanded = taus_expanded.permute(1,2,0)
        #print("DEBUG: taus_expanded: ", taus_expanded.size())

        #output are all zeros
        output = Variable(torch.zeros(input_reshape.size())).cuda()
        #max(0, z_i - taus), if values are negative, then it becomes zero!! (sparse!)
        #once we learn taus, sorted_z is not necessary.
        output = torch.max(output, input_shift - taus_expanded)
        #print("DEBUG: output: ", output.size())
        return output.view(-1, self.num_clusters*self.num_neurons_per_cluster), zs_sparse,taus, is_gt
		 
    # lots of in-place operations (e.g a += b) need manual gradient calculations...
    def backward(self, grad_output):
        self.output = self.output.view(-1,self.num_clusters, self.num_neurons_per_cluster)
        grad_output = grad_output.view(-1,self.num_clusters, self.num_neurons_per_cluster)
        dim = 2
        non_zeros = Variable(torch.ne(self.output, 0).type(torch.FloatTensor), requires_grad=False).cuda()
        mask_grad = Variable(torch.zeros(self.output.size()), requires_grad=False).cuda()
        mask_grad = torch.addcmul(mask_grad, non_zeros, grad_output)
        sum_mask_grad = torch.sum(mask_grad, dim)
        l1_norm_non_zeros = torch.sum(non_zeros, dim)
        sum_v = Variable(torch.zeros(sum_mask_grad.size()), requires_grad=False).cuda()
        sum_v = torch.addcdiv(sum_v, sum_mask_grad, l1_norm_non_zeros)
        self.gradInput = Variable(torch.zeros(grad_output.size()))
        self.gradInput = torch.addcmul(self.gradInput, non_zeros, grad_output - sum_v.expand_as(grad_output))
        self.gradInput = self.gradInput.view(-1, self.num_clusters*self.num_neurons_per_cluster)
        return self.gradInput

class MultiLabelSparseMaxLoss(nn.Module):

    def __init__(self, num_clusters, num_neurons_per_cluster):
        super(MultiLabelSparseMaxLoss, self).__init__()
        self.num_clusters = num_clusters
        self.num_neurons_per_cluster = num_neurons_per_cluster

    def forward(self, input, zs_sparse, target, output_sparsemax, taus, is_gt):
        self.output_sparsemax = output_sparsemax
        input = input.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        self.target = target.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        batch_size = input.size(0)
        dim = 2
        target_times_input = torch.sum(self.target * input, dim)
        target_inner_product = torch.sum(self.target * self.target, dim)
        zs_squared = zs_sparse * zs_sparse
        taus_squared = (taus * taus).expand_as(zs_squared.permute(2,0,1)).permute(1,2,0)
        #print("DEBUG: taus_squared: ", taus_squared.size())
        #print("DEBUG: is_gt: ", is_gt.size())
        taus_squared = taus_squared * is_gt
        sum_input_taus = torch.sum(zs_squared - taus_squared, dim) 
        sparsemax_loss = - target_times_input + 0.5*sum_input_taus + 0.5*target_inner_product
        sparsemax_loss = torch.sum(sparsemax_loss)/(batch_size * self.num_clusters)
        return sparsemax_loss

    def backward(self):
        grad_output = (- self.target + self.output_sparsemax)/(batch_size * self.num_clusters)
        return grad_output
