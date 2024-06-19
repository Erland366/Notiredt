
# Linear Kernel

I'll try to explain what's my understanding of the linear kernel here. Bare with me and please correct me if I understand it incorrectly.

## Difference between BLOCK and GROUP

I think one of the difference between `BLOCK` and `GROUP` is that, `BLOCK` is something that the compiler setting up for us, whereas `GROUP` is something that we setting up for us.

Because in Triton, we can just specify the `BLOCK` when we launching the kernel. Whereas `GROUP` is something that we're designing while we use those `pid` from the Triton blocks.

One thing to keep in mind is that `GROUP` is based on the `out_feat`, not anything else!

## Linear Kernel Forward

## Variables

`batch_dim` : My all number of batch

`BLOCK_SIZE_BATCH` : Simply BLOCK_SIZE on batch dimension
`n_batch_pids` :

- this is as simple as "how many block is needed in batch dim" based on the operation
- Just thinking of it as how many row needed for our operation
- We calculate this because batch dimension is easily parallelizable

`n_out_feat_pids` :

- This is as simple as "How many block is needed in out_feat dim" based on the operation
- Just thinking of it as how many column needed for our operation
- Based on that we calculate based on out_feat, then we basically will
    parallelize based on output_dim

`pids_per_group` : This is a number of `pids` on the group BUT based on the number of the `out_feat_dim`. Reminder that `GROUP` is calculated based on the `out_feat`

## How to come up with the formula?

`out_feat_pid`:

- First of all, recall that
