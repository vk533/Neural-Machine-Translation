# Neural Machine Translation
## Translating human readable dates into machine readable dates
The model built here could be used to translate from one language to another, such as translating from English to Hindi. However, language translation requires massive datasets and usually takes days of training on GPUs. But due to shortage of resources, a simpler 'date translation' task has been performed. The network will input a date written in a variety of possible formats (e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"). The network will translate them into standardized, machine readable dates (e.g. "1958-08-29", "1968-03-30", "1987-06-24"). We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD.

## Neural Machine Translation with attention
If you had to translate a book's paragraph from French to English, you would not read the whole paragraph, then close the book and translate. Even during the translation process, you would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English you are writing down. The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step.

### Attention mechanism
Here is a figure to remind you how the model works. The diagram on the left shows the attention model. The diagram on the right shows what one "Attention" step does to calculate the attention variables  α⟨t,t′⟩ , which are used to compute the context variable  context⟨t⟩  for each timestep in the output ( t=1,…,Ty ).

![Image](ImagesNMT/Picture1.png)

Here are some properties of the model that you may notice:

There are two separate LSTMs in this model (see diagram on the left). Because the one at the bottom of the picture is a Bi-directional LSTM and comes before the attention mechanism, we will call it pre-attention Bi-LSTM. The LSTM at the top of the diagram comes after the attention mechanism, so we will call it the post-attention LSTM. The pre-attention Bi-LSTM goes through  Tx  time steps; the post-attention LSTM goes through  Ty  time steps.

The post-attention LSTM passes  s⟨t⟩,c⟨t⟩  from one time step to the next. In the lecture videos, we were using only a basic RNN for the post-activation sequence model, so the state captured by the RNN output activations  s⟨t⟩ . But since we are using an LSTM here, the LSTM has both the output activation  s⟨t⟩  and the hidden cell state  c⟨t⟩ . However, unlike previous text generation examples (such as Dinosaurus in week 1), in this model the post-activation LSTM at time  t  does will not take the specific generated  y⟨t−1⟩  as input; it only takes  s⟨t⟩  and  c⟨t⟩  as input. We have designed the model this way, because (unlike language generation where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date.

We use  a⟨t⟩=[a→⟨t⟩;a←⟨t⟩]  to represent the concatenation of the activations of both the forward-direction and backward-directions of the pre-attention Bi-LSTM.

The diagram on the right uses a RepeatVector node to copy  s⟨t−1⟩ 's value  Tx  times, and then Concatenation to concatenate  s⟨t−1⟩  and  a⟨t⟩  to compute  e⟨t,t′ , which is then passed through a softmax to compute  α⟨t,t′⟩ 

### one_step_attention()
At step  t , given all the hidden states of the Bi-LSTM ( [a<1>,a<2>,...,a<Tx>] ) and the previous hidden state of the second LSTM ( s<t−1> ), one_step_attention() will compute the attention weights ( [α<t,1>,α<t,2>,...,α<t,Tx>] ) and output the context vector:
  
                                        context<t>=∑t′=0Txα<t,t′>a<t′>(1)
                                        
### model()
 Implements the entire model. It first runs the input through a Bi-LSTM to get back  [a<1>,a<2>,...,a<Tx>] . Then, it calls one_step_attention()  Ty  times (for loop). At each iteration of this loop, it gives the computed context vector  c<t>  to the second LSTM, and runs the output of the LSTM through a dense layer with softmax activation to generate a prediction  y^<t> .
  
Now you can use these layers  Ty  times in a for loop to generate the outputs, and their parameters will not be reinitialized. You will have to carry out the following steps:

- Propagate the input into a Bidirectional LSTM
- Iterate for  t=0,…,Ty−1 :
  1. Call one_step_attention() on  [α<t,1>,α<t,2>,...,α<t,Tx>]  and  s<t−1>  to get the context vector  context<t> .
  2. Give  context<t>  to the post-attention LSTM cell. Remember pass in the previous hidden-state  s⟨t−1⟩  and cell-states  c⟨t−1⟩     of this LSTM using initial_state= [previous hidden state, previous cell state]. Get back the new hidden state  s<t>  and the        new cell state  c<t> .
  3. Apply a softmax layer to  s<t> , get the output.
  4. Save the output by adding it to the list of outputs.
- Create your Keras model instance, it should have three inputs ("inputs",  s<0>  and  c<0> ) and output the list of "outputs".
