import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TextClassificationDataset
from models import BertClassifier
from process_data import load_data
from train_utils import save_checkpoint, load_checkpoint


def train_model(model, model_name, train_data, val_data, max_epochs=5, learning_rate=1e-6):
    # directory to save and load trained models
    models_directory = './models'

    # create DataSet and DataLoader objects
    train_ds, val_ds = TextClassificationDataset(train_data), TextClassificationDataset(val_data)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # try load checkpoint model and optimizer
    try:
        model, optimizer, train_loss_list, val_loss_list = load_checkpoint(model, optimizer, model_name, models_directory)
        print('Loaded pre-trained model.')
    except FileNotFoundError:
        # loss counters
        train_loss_list = []
        val_loss_list = []
        best_loss = torch.inf

    # epoch loop
    for epoch in range(max_epochs):

        # average train loss
        avg_train_loss = 0

        # training loop
        model.train()
        for train_input, train_label in tqdm(train_dl):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            # model output
            output = model(input_id, mask)

            # average train loss
            train_batch_loss = criterion(output, train_label.long())
            avg_train_loss += train_batch_loss.item() / len(train_dl)

            # optimizer step
            model.zero_grad()
            train_batch_loss.backward()
            optimizer.step()

        # save training loss
        train_loss_list.append(avg_train_loss)

        # evaluation loop
        model.eval()
        with torch.no_grad():
            # average validation loss
            avg_val_loss = 0
            for val_input, val_label in val_dl:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                # model output
                output = model(input_id, mask)

                # average validation loss
                val_batch_loss = criterion(output, val_label.long())
                avg_val_loss += val_batch_loss.item() / len(val_dl)

        # save validation metrics
        val_loss_list.append(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_checkpoint(model, optimizer, train_loss_list, val_loss_list, model_name, models_directory)

        # print progress
        print(f'Epoch [{epoch+1}/{max_epochs}], Train Loss: {round(avg_train_loss, 4)}, Valid Loss: {round(avg_val_loss, 4)}')



if __name__ == '__main__':
    # load train, validation, and test data
    train_data, val_data, test_data = load_data()

    # init BERT classifier as a baseline model
    bert_model = BertClassifier(n_classes=3)

    # train model
    train_model(bert_model, 'baseline_BERT', train_data, val_data)


