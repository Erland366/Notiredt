# Dropout Kernel

## Why it's divided by (1 - drop_p) if the value is not below the threshold

Because we need to stabilize all of the value so it won't get scaled down when inference. Remember that in inference, we will disable `dropout` completely. Hence, if we scaled down the value during training, it'll not stabilize during inference. Therefore, we should've scaled up the remaining values (values that's not zeroed) which is by simple inversing the remaining values.


