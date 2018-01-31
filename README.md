# aoa-visualization
An attention-over-attention visualization based on aoa implementation from
https://github.com/OlavHN/attention-over-attention

*model.py* is updated to match the newer interface of tensorflow (concat, stack, initializers, summary etc) and to output intermediate tensors such as attention values alpha, beta etc.

*attention.py* uses seaborn and pandas to plot attention values.

*reader.py* is updated to output a reverse index file for words
run with tokenize() line commented to only output index dictionaries.

*util.py* is also updated for tensorflow (name_scope)

#### To train a new model 
python model.py --training=True --name=my_model

#### To test accuracy 
python model.py --training=False --name=my_model --epochs=1 --dropout_keep_prob=1

#### To plot attention values
python attention.py --training=False --name=my_model --batch_size=8 --epochs=1 --dropout_keep_prob=1

This will only plot a single batch of batch_size samples, not all of the test data.

