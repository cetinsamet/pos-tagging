# pos-tagger
Implementation of a part-of-speech tagger for the English language  
A part-of-speech model is trained with a corpus containing almost 12,000 English sentences. 
  
Check for detailed Medium story that I wrote: https://medium.com/@cetinsamet/part-of-speech-pos-tagging-8af646a3d5bb  
  
## Model  
Model (mini model) in the repo is trained using mini data which contains only almost 3,000 English sentences.  
Therefore, it does not perform well enough.  
  
I encourage you to train a new model on your own using the corpus in data/ directory.  
My training lasted 32 minutes on a device with 3,1 GHz Intel Core i7 processor.  
Training accuracy was **100%** where development accuracy was **93%**.  
  
## Usage  
$**python3**  pos_tag.py  input-sentence  
  
### Example  
$**cd**  src  
$**python3** pos_tag.py  "Peace at home, peace in the world."  
**->** POS tagger is loaded.  
**->** [('Peace', 'NN'), ('at', 'DT'), ('home', 'NN'), (',', ','), ('peace', 'NN'), ('on', 'IN'), ('earth', 'NN'), ('.', '.')]    
  
*P.S. Above example is tested with a model trained on corpus in data/ directory(with almost 12,000 English sentences).*  
  
    
> Peace at home, peace in the world.  
> Mustafa Kemal ATATÜRK  
