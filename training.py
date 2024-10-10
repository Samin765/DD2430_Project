import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import utils, model_functions
import torch.nn as nn

class FinetuneCLIP():
  def __init__(self, dataloaders, clip, epochs = 200):
    self.dataloaders = dataloaders
    self.loss = {'train':[], 'val':[]}
    self.es = {'pat': 10, 'curr_pat' :0, 'min_loss': np.inf}# early stop
    self.conf ={'epochs': epochs }
    self.clip = clip # model and processor
    self.train_p = {} # Store trainable parameters here
    self.tt = {'soft':1, 'LoRA':0} # tuning method to use
    self.optimizer = None # config in initialize

  def train(self):
    """Training loop"""
    self.clip['m'].train()
    for epoch in tqdm(range(self.conf['epochs'])):
      running_loss, n_data = 0.0, 0
      for batch_nr, (image_embeds, labels, images) in enumerate(self.dataloaders['train']):
        self.optimizer.zero_grad()
        _, loss = self.forward(image_embeds, labels)
        loss.backward()
        self.optimizer.step()
        running_loss +=loss.item()
        n_data += len(labels)
        #print(self.train_p['soft'].grad)
      self.loss['train'].append(running_loss/n_data)
      if self.earlystop():
        self.load_p()# get best found
        return self.loss, self.train_p

  def forward(self, image_embeds, labels):
    """Get predictions of the model, add more here for different tuning methods"""
    train = True if image_embeds.shape[0]==len(labels) else False
    text = [self.train_p['add']+i for i in labels]
    #image_embeds, _ = get_image_emb(model, processor, return_normal(images, processor, 0, False)) #SLOW
    if self.tt['soft']:
      text_embeds = model_functions.get_text_emb_soft(self.clip['m'], self.clip['p'], text, self.train_p['soft'])
      logits_per_image, loss = model_functions.apply_clip(text_embeds, image_embeds, self.clip['m'], train=train)
    
    elif self.tt['LoRA']:
      pass
    else:
      text_embeds = model_functions.get_text_emb(self.clip['m'], self.clip['p'], text)
      logits_per_image, loss = model_functions.apply_clip(text_embeds, image_embeds, self.clip['m'], train=train)

    return logits_per_image, loss

  def eval(self, show_image=False):
    """Evaluate model on test set"""
    all_predictions, all_labels = [], []
    with torch.no_grad():
      for batch_nr, (image_embeds, labels, images) in enumerate(tqdm(self.dataloaders['test'])):
        logits_per_image, _ = self.forward(image_embeds, self.dataloaders['test'].dataset.classes)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        predicted_class = logits_per_image.argmax(dim=-1)
        all_predictions.append(predicted_class)
        for lab in labels:
            all_labels.append(self.dataloaders['test'].dataset.class_to_id[lab])
        if show_image and batch_nr%40==0:
            images = utils.return_normal(images, self.clip['p'], 4, True)
    all_predictions, all_labels=torch.cat(all_predictions).cpu(), torch.tensor(all_labels).cpu()
    acc = utils.accuracy(all_predictions, all_labels)
    print('Accuracy',acc)
    return all_predictions, all_labels, acc

  def earlystop(self):
    """Stop training when val loss start to increase"""
    with torch.no_grad():
      running_loss, n_data = 0.0, 0 # last batch can be smaller
      for batch_nr, (image_embeds, labels, images) in enumerate(self.dataloaders['val']):# val loader
          _, loss = self.forward(image_embeds, labels)
          running_loss +=loss.item()
          n_data += len(labels)
      self.loss['val'].append(running_loss/n_data)
      if len(self.loss['val'])>2:# early stop
          if self.es['curr_pat'] ==0:
              if running_loss> self.loss['val'][-2]: # if val_loss increase
                  self.es['min_loss'] = running_loss
                  torch.save(self.train_p['soft'], 'soft_prompts.pth')
                  self.es['curr_pat']+=1
          else:
              if running_loss> self.es['min_loss']: # if val_loss continute to increase
                  self.es['curr_pat']+=1
                  curr , pat = self.es['curr_pat'], self.es['pat']
                  print(f'Patience is {curr} / {pat}')
                  if self.es['curr_pat'] >=self.es['pat']:
                      return 'STOP'
              else: #reset
                  self.es['min_loss'] = np.inf
                  self.es['curr_pat'] = 0

  def plot_loss(self):
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, len(self.loss['train'])+1)), self.loss['train'], label='Training Loss')
    plt.plot(list(range(1, len(self.loss['val'])+1)), self.loss['val'], label='Validation Loss')
    # Adding labels and title
    plt.title('Loss Over Datapoints')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

  def load_p(self):
    """Load trained parameters"""
    self.train_p['soft']=torch.load('soft_prompts.pth', weights_only=True)

  def initialize(self, params):
    """Initialize trainable parameters"""
    self.train_p['add'] = params['add'] # the text added to classes

    if self.tt['soft']:
      self.train_p['soft'] = nn.Parameter(torch.zeros(params['num_soft'],
                self.clip['m'].text_projection.in_features), requires_grad=True)
      

      self.optimizer = torch.optim.Adam([self.train_p['soft']], lr=1e-3)
    if self.tt['LoRA']:
      pass
