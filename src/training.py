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
import time


class FinetuneCLIP():
    def __init__(self, dataloaders, clip, epochs=200):
        self.dataloaders = dataloaders
        # TODO decide if processor should have is_split_into_words = True
        self.clip = clip  # model and processor
        ## HARD MINING VARIABLES ##
        self.hard_mining = False #False --> Disable Hard Mining
        self.weighted = True
        self.hard_sample_threshold = 50 #loss for hard samples
        self.max_hard_samples = 500
        self.hard_samples = []  # Track hard samples
        ##########################
        self.loss = {'train': [], 'val': []}
        self.es = {'pat': 10, 'curr_pat': 0, 'min_loss': np.inf, 'best_model':clip['m']}  # early stop
        self.conf = {'epochs': epochs, 'balanced':True}
        self.train_p = {}  # Store trainable parameters here
        self.tt = {'soft': 1, 'LoRA': 0, 'image_fc': 0, 'desc': 0}  # tuning method to use
        self.optimizer = None  # config in initialize
        self.image_fc = None  # config in initialize
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_prefix = ""

    def train(self):
        """Training loop"""
        encoder = LabelEncoder()  
        Debug = False
        self.clip['m'].train()
        with tqdm(total=self.conf['epochs'], desc="Training", unit="epoch") as pbar:
            for epoch in range(self.conf['epochs']):
                max_loss_in_epoch = -9999
                epoch_hard_samples = []  # Initialize for each epoch
                # Print the learning rate and number of parameters in the optimizer
                lr = self.optimizer.param_groups[0]['lr']
                num_params = sum(p.numel() for p in self.optimizer.param_groups[0]['params'])
                ##DEBUG##
                # print(f"Epoch {epoch}: Optimizer - Learning Rate: {lr}, Number of Parameters: {num_params}")
                ##DEBUG##
                loss_values = []  # List to store loss values for the current epoch
                running_loss, n_data, n_data = 0.0, 0, 0
                if self.conf['balanced']:
                    for batch_nr, (image_embeds, labels, _, _) in enumerate(self.dataloaders['train']):
                        self.optimizer.zero_grad()
                        encoded_labels = encoder.fit_transform(labels)
                        batch_class_weights = self.get_class_weights(labels, encoded_labels)
                        _, loss = self.forward(image_embeds, labels, self.conf['balanced'], class_weights = batch_class_weights, encoded_labels = encoded_labels)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                else:
                    for batch_nr, (image_embeds, article_ids, feature, detail_desc) in enumerate(self.dataloaders['train']):
                         ##DEBUG##
                        if epoch > 20 and Debug:
                            if batch_nr % 10 == 0: 
                                print(f"Epoch {epoch}, Batch {batch_nr}: Memory Allocated = {torch.cuda.memory_allocated(self.device)}, Cached = {torch.cuda.memory_reserved(self.device)}")
                            if len(np.unique(feature)) != 18:
                                print(f"Epoch {epoch}, Batch {batch_nr}: Number of Unique Labels in Batch = {len(np.unique(feature))}")
                            ##DEBUG##
                        self.optimizer.zero_grad()
                        encoded_labels = encoder.fit_transform(feature)
                        batch_class_weights = self.get_class_weights(feature, encoded_labels)
                        _, loss = self.forward(image_embeds, labels=feature, balanced=self.conf['balanced'], class_weights=batch_class_weights, encoded_labels = encoded_labels)
                        # Hard Mining#
                        max_loss_in_epoch = max(max_loss_in_epoch, loss.item())  # Update max loss
                        loss_values.append(loss.item())
                        if loss.item() > self.hard_sample_threshold:  
                            epoch_hard_samples.append((image_embeds, feature))
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.detach().item()
                        del image_embeds, article_ids, feature, detail_desc, loss
                        torch.cuda.empty_cache()                  
                
                if self.hard_mining:
                    
                    if epoch % 10 == 0:
                        #self.hard_sample_threshold = threshold
                        loss_mean = np.mean(loss_values)
                        loss_std = np.std(loss_values)
                        threshold = loss_mean + 2 * loss_std
                        self.hard_sample_threshold = max_loss_in_epoch * 0.8
                        print(f"Suggested threshold for hard examples: {threshold}")
                        print(f"Epoch {epoch}: Max loss encountered = {max_loss_in_epoch}, Hard sample threshold set to {self.hard_sample_threshold}")
                    
                    if epoch > 10 and self.loss['val'][-1] > self.loss['val'][-2]:  # Increase if validation loss rises
                        self.max_hard_samples = min(len(self.dataloaders['train'].dataset), self.max_hard_samples + 20)
                    elif epoch % 10 == 0 and self.max_hard_samples > 50:  # Reduce periodically if high
                        self.max_hard_samples = max(50, self.max_hard_samples - 10)
                    #hard_example_count = sum(loss > self.hard_sample_threshold for loss in loss_values)
                    
                    self.hard_samples.extend(epoch_hard_samples)
                    self.hard_samples = self.hard_samples[-self.max_hard_samples:]  # Define max_hard_samples
                    if self.hard_samples:
                        print(f"Epoch {epoch}: Training on hard samples - Max allowed = {self.max_hard_samples}, Current count = {len(self.hard_samples)}")
                        self.train_on_hard_samples()
                    
                if epoch > 70 and epoch % 10 == 0:
                    if self.tt['LoRA']:
                        torch.save(
                            self.clip['m'].state_dict(), f'{self.model_prefix}_lora_model_{epoch}.pth'
                        )
                    print(f"Evaluating Model at Epoch {epoch}")
                    self.eval(encoded_labels)
                    all_predictions, all_labels, acc = self.eval(encoded_labels,False)
                    print(f"Accuracy of baseline is {acc:.2f}% at epoch {epoch}")
                    self.plot_loss_key('train', epoch)
                    self.plot_loss_key('val', epoch)

                self.loss['train'].append(running_loss/len(self.dataloaders['train']))
                if self.earlystop(encoded_labels, epoch):
                    print(f"Early Stopping Triggered at Epoch {epoch}, Loading Best Model")
                    self.load_p()  # get parameters best found
                    return self.loss, self.train_p
                pbar.set_postfix({"Patience": f"{self.es['curr_pat']} / {self.es['pat']}"})
                pbar.update(1)
            return self.loss, self.train_p
        
    def train_on_hard_samples(self):
        self.clip['m'].train()
        #print(f"Training on hard samples: Max allowed = {self.max_hard_samples}, Current count = {len(self.hard_samples)}")
        running_loss = 0.0
        for image_embeds, labels in self.hard_samples:
            self.optimizer.zero_grad()
            encoded_labels = LabelEncoder().fit_transform(labels)
            batch_class_weights = self.get_class_weights(labels, encoded_labels)
            logits, loss = self.forward(image_embeds, labels, self.conf['balanced'], class_weights=batch_class_weights, encoded_labels=encoded_labels)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            #print(labels)


    def forward(self, image_embeds, labels, balanced, class_weights = None, descriptions=None ,encoded_labels = None):
        """Get predictions of the model, add more here for different tuning methods"""
        train = True if image_embeds.shape[0] == len(labels) else False
        texts = []
        
        if descriptions and self.tt['desc']:
            for desc in descriptions:
                new_text =\
                    [f'An image of clothing with name: {label}, and description: {desc}' for label in labels]
                texts.append(new_text)
        else:
            texts = [self.train_p['add']+i for i in labels]

        # image_embeds, _ = get_image_emb(model, processor, return_normal(images, processor, 0, False)) #SLOW

        # image_fc is just adding a fc layer to the image embeddings
        # so we can do that before soft and lora because they are only applied to text
        if self.tt['image_fc']:
            image_embeds = self.image_fc(image_embeds)
        if self.tt['soft']:
            text_embeds = model_functions.get_text_emb_soft(
                self.clip['m'], self.clip['p'], texts, self.train_p['soft'])
        elif self.tt['desc']:
            assert descriptions is not None
            embs = []
            for text in texts:
                embs.append(model_functions.get_text_emb(
                    self.clip['m'], self.clip['p'], text))
            text_embeds = torch.cat(embs)
        else:
            text_embeds = model_functions.get_text_emb(
                self.clip['m'], self.clip['p'], texts)

        logits_per_image, loss = model_functions.apply_clip(
            text_embeds, image_embeds,
            self.clip['m'], balanced=balanced,
            train=train, labels=labels,
            class_weights=class_weights,
            encoded_labels = encoded_labels,
            weighted=self.weighted
        )

        return logits_per_image, loss

    def eval(self, encoded_labels = None, show_image=False):
        """Evaluate model on test set"""
        encoder = LabelEncoder() 
        all_predictions, all_labels = [], []
        with torch.no_grad():
            if self.conf['balanced']:
                for batch_nr, (image_embeds, labels, _, detail_desc) in enumerate(tqdm(self.dataloaders['test'])):
                    encoded_labels = encoder.fit_transform(flabels)
                    batch_class_weights = self.get_class_weights(labels, encoded_labels)
                    logits_per_image, _ = self.forward(
                        image_embeds,
                        self.dataloaders['test'].dataset.classes,
                        self.conf['balanced'],
                        class_weights=batch_class_weights,
                        descriptions=detail_desc
                        )

                    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    predicted_class = logits_per_image.argmax(dim=-1)
                    all_predictions.append(predicted_class)
                    for lab in labels:
                        all_labels.append(
                            self.dataloaders['test'].dataset.class_to_id[lab])
            else:
                for batch_nr, (image_embeds, article_ids, feature, detail_desc) in enumerate(tqdm(self.dataloaders['test'])):
                    encoded_labels = encoder.fit_transform(feature)
                    batch_class_weights = self.get_class_weights(feature, encoded_labels)
                    logits_per_image, _ = self.forward(
                        image_embeds,
                        self.dataloaders['test'].dataset.classes,
                        self.conf['balanced'],
                        class_weights=batch_class_weights,
                        descriptions=detail_desc,
                        encoded_labels = encoded_labels
                        )

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

    def earlystop(self, encoded_labels = None, epoch=None):
        encoder = LabelEncoder() 
        """Stop training when val loss start to increase"""
        with torch.no_grad():
            running_loss = 0.0  # last batch can be smaller
            # val loader
            if self.conf['balanced']:
                for batch_nr, (image_embeds, labels, _,_) in enumerate(self.dataloaders['val']):
                    #_, loss = self.forward(image_embeds, labels)
                    encoded_labels = encoder.fit_transform(labels)
                    batch_class_weights = self.get_class_weights(labels, encoded_labels)
                    _, loss = self.forward(image_embeds, labels, self.conf['balanced'], class_weights = batch_class_weights, encoded_labels = encoded_labels)

                    running_loss += loss.item()
            else:
                for batch_nr, (image_embeds, article_ids, feature, detail_desc) in enumerate(self.dataloaders['train']):
                    #_, loss = self.forward(image_embeds, feature)
                    encoded_labels = encoder.fit_transform(feature)
                    batch_class_weights = self.get_class_weights(feature, encoded_labels)
                    _, loss = self.forward(image_embeds, feature, self.conf['balanced'], class_weights = batch_class_weights, encoded_labels = encoded_labels)
                    running_loss += loss.item()

            self.loss['val'].append(running_loss/len(self.dataloaders['val']))
            # write loss to file
            with open(f'{self.model_prefix}_loss.txt', 'a') as f:
                f.write(f'{running_loss/len(self.dataloaders["val"])}\n')

            if len(self.loss['val']) > 2:  # early stop
                if self.es['curr_pat'] == 0:
                    # if val_loss increase first time
                    if running_loss > self.loss['val'][-2]:
                        self.es['min_loss'] = running_loss
                        self.es['best_model'] = copy.deepcopy(self.clip['m'])
                        if self.tt['soft']:
                            torch.save(
                                self.train_p['soft'], f'{self.model_prefix}_soft.pth')
                        if self.tt['LoRA']:
                            torch.save(
                                self.clip['m'].state_dict(), f'{self.model_prefix}_lora.pth'
                            )
                        
                        self.es['curr_pat'] += 1
                else:
                    # if val_loss continute to increase
                    if running_loss > self.es['min_loss']:
                        self.es['curr_pat'] += 1
                        #curr, pat = self.es['curr_pat'], self.es['pat']
                        #print(f'Patience is {curr} / {pat}')
                        if self.es['curr_pat'] >= self.es['pat']:
                            print(f"Early Stopping Triggered at Epoch {epoch}")
                            return 'STOP'
                    else:  # reset
                        #self.es['min_loss'] = np.inf
                        self.es['min_loss'] = running_loss
                        self.es['curr_pat'] = 0
                        # print(f"Patience reset, New Min Loss = {self.es['min_loss']}")


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
        
    def plot_loss_key(self, key, epoch):
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
        plt.savefig(f'{self.model_prefix}_{key}_loss_{epoch}.png')


    def load_p(self, file_name=None):
        """Load trained parameters, add more here"""
        self.clip['m'] = self.es['best_model'] # the parameters at minimum
        if self.tt['soft']:
            file_name = f'{self.model_prefix}_soft.pth' if file_name is None else file_name
            self.train_p['soft'] = torch.load(file_name)
        elif self.tt['LoRA']:
            file_name = f'{self.model_prefix}_lora.pth' if file_name is None else file_name
            self.clip['m'].load_state_dict(torch.load(file_name))
        else:
            raise Exception('Need to specify file_name or have a tuning method')

    def initialize(self, params, load=False, file_name=None):
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
            if load:
                self.load_p(file_name=file_name) # load stored parameters
                lora_params_attention = model_functions.get_lora_params(
                   self.clip['m'], print_layer=False
                )
                tunable_params += list(lora_params_attention)
            else:
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

    def get_class_weights(self,labels , encoded_labels):
        if not self.conf['balanced']:
            #encoder = LabelEncoder()
            #encoded_labels = encoder.fit_transform(labels)
            encoded_labels_tensor = torch.tensor(encoded_labels)
            classes = torch.unique(encoded_labels_tensor)
            class_weights = compute_class_weight(class_weight = 'balanced' , classes = classes.cpu().numpy(), y = encoded_labels)
            class_weights = torch.tensor(class_weights, dtype = torch.float32, device= encoded_labels_tensor.device)
        else:
            class_weights = None
        return class_weights
