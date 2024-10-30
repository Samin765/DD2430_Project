import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import utils
import model_functions
import torch.nn as nn
import copy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


class FinetuneCLIP():
    def __init__(self, dataloaders, clip, epochs=200):
        self.dataloaders = dataloaders
        self.clip = clip  # model and processor
        self.loss = {'train': [], 'val': []}
        self.es = {'pat': 10, 'curr_pat': 0, 'min_loss': np.inf, 'best_model':clip['m']}  # early stop
        self.conf = {'epochs': epochs, 'balanced':True}
        self.train_p = {}  # Store trainable parameters here
        self.tt = {'soft': 1, 'LoRA': 0, 'image_fc': 0}  # tuning method to use
        self.optimizer = None  # config in initialize
        self.image_fc = None  # config in initialize
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        """Training loop"""
        self.clip['m'].train()
        with tqdm(total=self.conf['epochs'], desc="Training", unit="epoch") as pbar:
            for epoch in range(self.conf['epochs']):
                running_loss, n_data, n_data = 0.0, 0, 0
                if self.conf['balanced']:
                    for batch_nr, (image_embeds, labels, _, _) in enumerate(self.dataloaders['train']):
                        self.optimizer.zero_grad()
                        batch_class_weights = self.get_class_weights(labels)
                        _, loss = self.forward(image_embeds, labels, self.conf['balanced'], class_weights = batch_class_weights)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                else:
                    for batch_nr, (image_embeds, article_ids, feature, detail_desc) in enumerate(self.dataloaders['train']):
                        
                        self.optimizer.zero_grad()
                        
                   
                        batch_class_weights = self.get_class_weights(feature)
                        #print(class_weights)
                        _, loss = self.forward(image_embeds, feature, self.conf['balanced'], class_weights = batch_class_weights)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.detach().item()
                        del image_embeds, article_ids, feature, detail_desc, loss
                        torch.cuda.empty_cache()  
                    
                    
                self.loss['train'].append(running_loss/len(self.dataloaders['train']))
                if self.earlystop():
                    self.load_p()  # get parameters best found
                    return self.loss, self.train_p
                pbar.set_postfix({"Patience": f"{self.es['curr_pat']} / {self.es['pat']}"})
                pbar.update(1)
            return self.loss, self.train_p
                

    def forward(self, image_embeds, labels, balanced, class_weights = None):
        """Get predictions of the model, add more here for different tuning methods"""
        train = True if image_embeds.shape[0] == len(labels) else False
        text = [self.train_p['add']+i for i in labels]
        # image_embeds, _ = get_image_emb(model, processor, return_normal(images, processor, 0, False)) #SLOW

        # image_fc is just adding a fc layer to the image embeddings
        # so we can do that before soft and lora because they are only applied to text
        if self.tt['image_fc']:
            image_embeds = self.image_fc(image_embeds)

        if self.tt['soft']:
            text_embeds = model_functions.get_text_emb_soft(
                self.clip['m'], self.clip['p'], text, self.train_p['soft'])
            logits_per_image, loss = model_functions.apply_clip(
                text_embeds, image_embeds, self.clip['m'],balanced, train=train, labels = labels, class_weights = class_weights)
        else:
            text_embeds = model_functions.get_text_emb(
                self.clip['m'], self.clip['p'], text)
            logits_per_image, loss = model_functions.apply_clip(
                text_embeds, image_embeds, self.clip['m'], balanced = balanced, train=train, labels = labels, class_weights = class_weights)
        return logits_per_image, loss

    def eval(self, show_image=False):
        """Evaluate model on test set"""
        all_predictions, all_labels = [], []
        with torch.no_grad():
            if self.conf['balanced']:
                for batch_nr, (image_embeds, labels, _, _) in enumerate(tqdm(self.dataloaders['test'])):
                    #logits_per_image, _ = self.forward(
                    #    image_embeds, self.dataloaders['test'].dataset.classes)
                    batch_class_weights = self.get_class_weights(labels)
                    logits_per_image, _ = self.forward(image_embeds, self.dataloaders['test'].dataset.classes, self.conf['balanced'], class_weights = batch_class_weights)

                    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    predicted_class = logits_per_image.argmax(dim=-1)
                    all_predictions.append(predicted_class)
                    for lab in labels:
                        all_labels.append(
                            self.dataloaders['test'].dataset.class_to_id[lab])
            else:
                for batch_nr, (image_embeds, article_ids, feature, detail_desc) in enumerate(tqdm(self.dataloaders['test'])):
                    #logits_per_image, _ = self.forward(
                    #    image_embeds, self.dataloaders['test'].dataset.classes)
                    batch_class_weights = self.get_class_weights(feature)
                    logits_per_image, _ = self.forward(image_embeds, self.dataloaders['test'].dataset.classes, self.conf['balanced'], class_weights = batch_class_weights)

                    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    predicted_class = logits_per_image.argmax(dim=-1)
                    all_predictions.append(predicted_class)
                    for lab in feature:
     
                        all_labels.append(
                            self.dataloaders['test'].dataset.class_to_id[lab])
                   
        all_predictions, all_labels = torch.cat(
            all_predictions).cpu(), torch.tensor(all_labels).cpu()
        acc = utils.accuracy(all_predictions, all_labels)
        print('Accuracy', acc)
        return all_predictions, all_labels, acc

    def earlystop(self):
        """Stop training when val loss start to increase"""
        with torch.no_grad():
            running_loss = 0.0  # last batch can be smaller
            # val loader
            if self.conf['balanced']:
                for batch_nr, (image_embeds, labels, _,_) in enumerate(self.dataloaders['val']):
                    #_, loss = self.forward(image_embeds, labels)
                    batch_class_weights = self.get_class_weights(labels)
                    _, loss = self.forward(image_embeds, labels, self.conf['balanced'], class_weights = batch_class_weights)

                    running_loss += loss.item()
            else:
                for batch_nr, (image_embeds, article_ids, feature, detail_desc) in enumerate(self.dataloaders['train']):
                    #_, loss = self.forward(image_embeds, feature)
                    batch_class_weights = self.get_class_weights(feature)
                    _, loss = self.forward(image_embeds, feature, self.conf['balanced'], class_weights = batch_class_weights)
                    running_loss += loss.item()

            self.loss['val'].append(running_loss/len(self.dataloaders['val']))
            if len(self.loss['val']) > 2:  # early stop
                if self.es['curr_pat'] == 0:
                    # if val_loss increase first time
                    if running_loss > self.loss['val'][-2]:
                        self.es['min_loss'] = running_loss
                        self.es['best_model'] = copy.deepcopy(self.clip['m'])
                        if self.tt['soft']:
                            torch.save(
                                self.train_p['soft'], 'soft_prompts.pth')
                        
                        self.es['curr_pat'] += 1
                else:
                    # if val_loss continute to increase
                    if running_loss > self.es['min_loss']:
                        self.es['curr_pat'] += 1
                        #curr, pat = self.es['curr_pat'], self.es['pat']
                        #print(f'Patience is {curr} / {pat}')
                        if self.es['curr_pat'] >= self.es['pat']:
                            return 'STOP'
                    else:  # reset
                        self.es['min_loss'] = np.inf
                        self.es['curr_pat'] = 0

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(1, len(self.loss['train'])+1)),
                 self.loss['train'], label='Training Loss')
        plt.plot(list(range(1, len(self.loss['val'])+1)),
                 self.loss['val'], label='Validation Loss')
        
        # Adding labels and title
        plt.title('Loss Over Datapoints')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_loss_key(self, key):
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(1, len(self.loss[key])+1)),
                 self.loss[key], label=f'{key} Loss')
        # Adding labels and title
        plt.title('Loss Over Datapoints')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def load_p(self):
        """Load trained parameters, add more here"""
        self.clip['m'] = self.es['best_model'] # the parameters at minimum
        if self.tt['soft']:
            self.train_p['soft'] = torch.load(
                'soft_prompts.pth', weights_only=True)
        

    def initialize(self, params, load=False):
        """Initialize trainable parameters"""
        added_text = params.get('add', '')

        self.train_p['add'] = added_text  # the text added to classes
        tunable_params = []
        # default values are same as pytorch adam default
        lr = params.get('lr', 1e-3)
        weight_decay = params.get('weight_decay', 0)

        if self.tt['soft']:
            if not load:
                self.train_p['soft'] = nn.Parameter(torch.zeros(params['num_soft'],
                                                    self.clip['m'].text_projection.in_features), requires_grad=True)
                tunable_params.append(self.train_p['soft'])
                assert self.train_p['soft'].is_leaf == tunable_params[0].is_leaf
            else:
                self.load_p() # load stored parameters
                tunable_params.append(self.train_p['soft'])

        if self.tt['image_fc']:
            self.image_fc = nn.Linear(512, 512).to(self.device)
            tunable_params += list(self.image_fc.parameters())

        if self.tt['LoRA']:
            self.train_p['LoRA'] = params['LoRA']
            tunable_params += list(params['LoRA'])

        # Add more options here if you need to

        if tunable_params:
            self.optimizer = torch.optim.Adam(
                tunable_params, lr=lr, weight_decay=weight_decay)

    def count_parameters(self):
        if self.optimizer is None or not self.optimizer.param_groups:
            print("Optimizer has no parameters")
            return

        num_params = sum(p.numel()
                         for p in self.optimizer.param_groups[0]['params'])
        print(f'Total number of parameters in the optimizer: {num_params}')


    def get_class_weights(self,labels):
        if not self.conf['balanced']:
            encoder = LabelEncoder()
            encoded_labels = encoder.fit_transform(labels)
            encoded_labels_tensor = torch.tensor(encoded_labels)
            classes = torch.unique(encoded_labels_tensor)
            class_weights = compute_class_weight(class_weight = 'balanced' , classes = classes.cpu().numpy(), y = encoded_labels)
            class_weights = torch.tensor(class_weights, dtype = torch.float32, device= encoded_labels_tensor.device)
        else:
            class_weights = None
        return class_weights
