import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import utils
import model_functions
import torch.nn as nn


class FinetuneCLIP():
    def __init__(self, dataloaders, clip, epochs=200):
        self.dataloaders = dataloaders
        self.loss = {'train': [], 'val': []}
        self.es = {'pat': 10, 'curr_pat': 0, 'min_loss': np.inf}  # early stop
        self.conf = {'epochs': epochs}
        self.clip = clip  # model and processor
        self.train_p = {}  # Store trainable parameters here
        self.tt = {'soft': 1, 'LoRA': 0, 'image_fc': 0}  # tuning method to use
        self.optimizer = None  # config in initialize
        self.image_fc = None  # config in initialize
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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
                running_loss += loss.item()
                n_data += len(labels)
                # print(self.train_p['soft'].grad)
            self.loss['train'].append(running_loss/n_data)
            if self.earlystop():
                if self.tt['soft']:
                    self.load_p()  # get best found
                    return self.loss, self.train_p

                return self.loss

    def forward(self, image_embeds, labels):
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
                text_embeds, image_embeds, self.clip['m'], train=train)
        else:
            text_embeds = model_functions.get_text_emb(
                self.clip['m'], self.clip['p'], text)
            logits_per_image, loss = model_functions.apply_clip(
                text_embeds, image_embeds, self.clip['m'], train=train)

        return logits_per_image, loss

    def eval(self, show_image=False):
        """Evaluate model on test set"""
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for batch_nr, (image_embeds, labels, images) in enumerate(tqdm(self.dataloaders['test'])):
                logits_per_image, _ = self.forward(
                    image_embeds, self.dataloaders['test'].dataset.classes)
                # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                predicted_class = logits_per_image.argmax(dim=-1)
                all_predictions.append(predicted_class)
                for lab in labels:
                    all_labels.append(
                        self.dataloaders['test'].dataset.class_to_id[lab])
                if show_image and batch_nr % 40 == 0:
                    images = utils.return_normal(
                        images, self.clip['p'], 4, True)
        all_predictions, all_labels = torch.cat(
            all_predictions).cpu(), torch.tensor(all_labels).cpu()
        acc = utils.accuracy(all_predictions, all_labels)
        print('Accuracyasdasdas', acc)
        return all_predictions, all_labels, acc

    def earlystop(self):
        """Stop training when val loss start to increase"""
        with torch.no_grad():
            running_loss, n_data = 0.0, 0  # last batch can be smaller
            # val loader
            for batch_nr, (image_embeds, labels, images) in enumerate(self.dataloaders['val']):
                _, loss = self.forward(image_embeds, labels)
                running_loss += loss.item()
                n_data += len(labels)
            self.loss['val'].append(running_loss/n_data)
            if len(self.loss['val']) > 2:  # early stop
                if self.es['curr_pat'] == 0:
                    # if val_loss increase
                    if running_loss > self.loss['val'][-2]:
                        self.es['min_loss'] = running_loss
                        if self.tt['soft']:
                            torch.save(
                                self.train_p['soft'], 'soft_prompts.pth')
                        self.es['curr_pat'] += 1
                else:
                    # if val_loss continute to increase
                    if running_loss > self.es['min_loss']:
                        self.es['curr_pat'] += 1
                        curr, pat = self.es['curr_pat'], self.es['pat']
                        print(f'Patience is {curr} / {pat}')
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

    def load_p(self):
        """Load trained parameters"""
        self.train_p['soft'] = torch.load(
            'soft_prompts.pth', weights_only=True)

    def initialize(self, params):
        """Initialize trainable parameters"""
        added_text = params.get('add', '')

        self.train_p['add'] = added_text  # the text added to classes
        tunable_params = []
        # default values are same as pytorch adam default
        lr = params.get('lr', 1e-3)
        weight_decay = params.get('weight_decay', 0)

        if self.tt['soft']:
            self.train_p['soft'] = nn.Parameter(torch.zeros(params['num_soft'],
                                                self.clip['m'].text_projection.in_features), requires_grad=True)
            tunable_params += list(self.train_p['soft'])

        if self.tt['image_fc']:
            self.image_fc = nn.Linear(512, 512).to(self.device)
            tunable_params += list(self.image_fc)

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
