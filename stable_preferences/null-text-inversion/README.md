# Null text inversion
In this [notebook](run_null_text.ipynb) you can find the code for null-textual inversion, you just need to change the device configuration in base on what you would like to use. 

To avoid issues while running it, be sure to satisfy the requirement present in `requirement.txt`.

Sometimes it complaines on the unet forward for the presence of `encoder_hidden_states` argument. In that case you should make sure to have moved the unet to a gpu device (at least it was my way of solving this problem).