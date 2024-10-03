import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _create_4d_causal_attention_mask
from transformers.models.clip.modeling_clip import clip_loss

# Needed to create the dataset
def get_image_emb(model, processor, images):
    """Given an tensor of batch images returns the batch image embeddings [batch, 512]"""

    #print(model.vision_model, processor.image_processor)
    vision_model = model.vision_model # VIT
    image_processor = processor.image_processor # standardise the input
    visual_projection = model.visual_projection # fc layer

    #standardise, same shape as image
    prosessed_images = image_processor(images, return_tensors='pt')['pixel_values']

    # apply VIT snd project to latent space  dim [batch, 768]
    vision_latent = vision_model(prosessed_images.to(model.device))[1] # not same as text

    # project to same dim as text emb by FC layer [batch, 512]
    image_embeds = visual_projection(vision_latent)

    # normalize so norm is one, good for dot product later
    return image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True), prosessed_images

def get_text_emb(model, processor, text):
    """Given an tensor of batch text returns the batch text embeddings [batch, 512],
    define X as the number of tokens and might differ from text length"""
    #print(model.text_model, processor.tokenizer)
    text_model = model.text_model # VIT
    text_tokenizer = processor.tokenizer # tokenize the input
    text_projection = model.text_projection # fc layer
    #tokenize, returns 2 tensors, tokens and attention mask [batch, X]
    tokenized_text = text_tokenizer(text, return_tensors='pt', padding=True, truncation = True)
    # apply TRANSFORMER and project to latent space  dim [batch, 512]
    text_latent = text_model(**tokenized_text.to(model.device))[1]
    # project to same dim as text emb by FC layer [batch, 512] unneccessary???
    text_embeds = text_projection(text_latent) #[batch, 512] to [batch, 512] same
    # TODO add LORA
    # normalize so norm is one, good for dot product later
    return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

def apply_clip(text_embeds, image_embeds, model, train=False):
    """Forward pass of clip"""
    device = model.device # use gpu
    logit_scale = model.logit_scale.exp().to(device) # temperature param
    text_embeds, image_embeds = text_embeds.to(device), image_embeds.to(device)
    # TODO add LORA
    logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
    loss = 0
    if train: # must have same ammount of text as images for training
        loss = clip_loss(logits_per_image.t())
    return logits_per_image, loss

def get_text_emb_soft(model, processor, text, soft_prompt_hidden):
    """Just like get_text_emb but for sof prompts,
    define X as the number of tokens and might differ from text length"""
    device = model.device
    #print(model.text_model, processor.tokenizer)
    text_model = model.text_model # VIT original
    text_tokenizer = processor.tokenizer # tokenize the input
    text_projection = model.text_projection # fc layer
    text_embedder_inner = text_model.embeddings # OBS this is the inner embedding NOT the one we want
    # tokenize the text, returns 2 tensors, tokens and attention mask [batch,X+soft]# token len not same as text
    tokenized_text = text_tokenizer(text, return_tensors='pt', padding=True, truncation = True) # returns tokens and attention mask
    # add soft prompt--------------
    # Take out the parts
    input_ids, attention_mask = tokenized_text['input_ids'].to(device), tokenized_text['attention_mask'].to(device)
    attention_mask = attention_mask
    # get only hiddden states, this is before textTransformer is applied
    hidden_states= text_embedder_inner(input_ids) #torch.Size([batch_size, X, 512])
    batch_size = hidden_states.size(0)
    # adding vectors to the embedding torch.Size([4, X+softprompts, 512])
    expand_hidden = soft_prompt_hidden.unsqueeze(0).expand(batch_size, -1, -1)
    hidden_states = torch.cat([expand_hidden.to(device), hidden_states], dim=1)
    # must match the shape
    soft_prompt_attention_mask = torch.ones(batch_size, soft_prompt_hidden.shape[0], dtype=attention_mask.dtype)
    attention_mask = torch.cat([soft_prompt_attention_mask.to(device), attention_mask], dim=1)# just ones
    #end of soft prompt--------------
    # apply costum transformer snd project to latent space  dim [batch, 512]
    text_latent = forward_text(input_ids, attention_mask, hidden_states, text_model)
    # project to same dim as text emb by FC layer [batch, 512] unneccessary???
    text_embeds = text_projection(text_latent) #[batch, 512] to [batch, 512] same
    # TODO add LORA
    # normalize so norm is one, good for dot product later
    return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

def get_text_emb_soft_loralt(model, processor, text, soft_prompt_hidden, text_lora_layer):
    """Just like get_text_emb but for sof prompts,
    define X as the number of tokens and might differ from text length"""
    #print(model.text_model, processor.tokenizer)
    device = model.device
    text_model = model.text_model # VIT original
    text_tokenizer = processor.tokenizer # tokenize the input
    text_projection = model.text_projection # fc layer
    text_embedder_inner = text_model.embeddings # OBS this is the inner embedding NOT the one we want
    # tokenize the text, returns 2 tensors, tokens and attention mask [batch,X+soft]# token len not same as text
    tokenized_text = text_tokenizer(text, return_tensors='pt', padding=True, truncation = True) # returns tokens and attention mask
    # add soft prompt--------------
    # Take out the parts
    input_ids, attention_mask = tokenized_text['input_ids'].to(device), tokenized_text['attention_mask'].to(device)
    attention_mask = attention_mask
    # get only hiddden states, this is before textTransformer is applied
    hidden_states= text_embedder_inner(input_ids) #torch.Size([batch_size, X, 512])
    batch_size = hidden_states.size(0)
    # adding vectors to the embedding torch.Size([4, X+softprompts, 512])
    expand_hidden = soft_prompt_hidden.unsqueeze(0).expand(batch_size, -1, -1)
    hidden_states = torch.cat([expand_hidden.to(device), hidden_states], dim=1)
    # must match the shape
    soft_prompt_attention_mask = torch.ones(batch_size, soft_prompt_hidden.shape[0], dtype=attention_mask.dtype)
    attention_mask = torch.cat([soft_prompt_attention_mask.to(device), attention_mask], dim=1)# just ones
    #end of soft prompt--------------
    # apply costum transformer snd project to latent space  dim [batch, 512]
    text_latent = forward_text(input_ids, attention_mask, hidden_states, text_model)
    # project to same dim as text emb by FC layer [batch, 512] unneccessary???

    # Almost same thing as adding Lora to Last Transform Layer in projection attention layer
    # we can remove this if we want and just add it in the loop when creating Lora layers for the attention layers
    text_latent = text_lora_layer(text_latent)
    text_embeds = text_projection(text_latent) #[batch, 512] to [batch, 512] same

    # TODO add LORA
    # normalize so norm is one, good for dot product later
    return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

def forward_text(input_ids, attention_mask, hidden_states, text_model):
    """Modified forward pass of the text model TRANSFORMER to include soft prompts"""
    #print(text_model) # prints the architecture
    num_soft = hidden_states.shape[1]-input_ids.shape[1]
    input_shape = input_ids.size()

    causal_attention_mask = _create_4d_causal_attention_mask((hidden_states.shape[0], hidden_states.shape[1])
    , hidden_states.dtype, device=hidden_states.device)
    if attention_mask is not None and not text_model._use_flash_attention_2:
        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask)
    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_model.final_layer_norm(last_hidden_state)
    last_hidden_state = last_hidden_state[:, num_soft:, :] # remove prompt states
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.view(-1, input_shape[-1]).to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),]
    return pooled_output