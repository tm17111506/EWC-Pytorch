from torch import optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from visdom import Visdom
from datetime import date

import utils
import visual
import os
import pickle
import torch
from pdb import set_trace as bp

def train(model, train_datasets, test_datasets, epochs_per_task=10,
          batch_size=64, test_size=1024, consolidate=True,
          fisher_estimation_sample_size=1024,
          lr=1e-3, weight_decay=1e-5,
          loss_log_interval=30,
          eval_log_interval=50,
          grad_log_interval=50,
          cuda=False,
          writer=None,
          args=None):
    # prepare the loss criteriton and the optimizer.
    criteriton = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          weight_decay=weight_decay)

    # instantiate a visdom client
    # vis = Visdom(env=model.name)

    # set the model's mode to training mode.
    model.train()

    for task, train_dataset in enumerate(train_datasets, 1):
        for epoch in range(1, epochs_per_task+1):
            # prepare the data loaders.
            data_loader = utils.get_data_loader(
                train_dataset, batch_size=batch_size,
                cuda=cuda
            )
            data_stream = tqdm(enumerate(data_loader, 1))

            for batch_index, (x, y) in data_stream:
                # where are we?
                data_size = len(x)
                dataset_size = len(data_loader.dataset)
                dataset_batches = len(data_loader)
                previous_task_iteration = sum([
                    epochs_per_task * len(d) // batch_size for d in
                    train_datasets[:task-1]
                ])
                current_task_iteration = (
                    (epoch-1)*dataset_batches + batch_index
                )
                iteration = (
                    previous_task_iteration +
                    current_task_iteration
                )

                # prepare the data.
                x = x.view(data_size, -1)
                x = Variable(x).cuda() if cuda else Variable(x)
                y = Variable(y).cuda() if cuda else Variable(y)

                # run the model and backpropagate the errors.
                optimizer.zero_grad()
                scores = model(x)
                ce_loss = criteriton(scores, y)
                ewc_loss = model.ewc_loss(cuda=cuda)
                loss = ce_loss + ewc_loss
                loss.backward()
                optimizer.step()

                # calculate the training precision.
                _, predicted = scores.max(1)
                precision = (predicted == y).sum().float() / len(x)

                data_stream.set_description((
                    '=> '
                    'task: {task}/{tasks} | '
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'prec: {prec:.4} | '
                    'loss => '
                    'ce: {ce_loss:.4} / '
                    'ewc: {ewc_loss:.4} / '
                    'total: {loss:.4}'
                ).format(
                    task=task,
                    tasks=len(train_datasets),
                    epoch=epoch,
                    epochs=epochs_per_task,
                    trained=batch_index*batch_size,
                    total=dataset_size,
                    progress=(100.*batch_index/dataset_batches),
                    prec=float(precision),
                    ce_loss=float(ce_loss),
                    ewc_loss=float(ewc_loss),
                    loss=float(loss),
                ))

                # Store the gradients during training process
                # if iteration % grad_log_interval == 0:
                # param_dict = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_scalar(name, torch.mean(param.grad), iteration)
                #         param_dict[name] = param.grad.cpu().detach().data.numpy()

                # if args != None:
                #     param_dict_path = args.grad_log_path
                # param_dict_path = param_dict_path + date.today().strftime('%m-%d-%y') \
                #                     + os.sep + 'task_{}'.format(task)
                # if not os.path.exists(param_dict_path):
                #     os.makedirs(param_dict_path)

                # with open(param_dict_path + os.sep + '{}.pkl'.format(iteration), 'wb+') as f:
                #     pickle.dump(param_dict, f)


                # Send test precision to the visdom server.
                # if iteration % eval_log_interval == 0:
                #     names = [
                #         'task {}'.format(i+1) for i in
                #         range(len(train_datasets))
                #     ]
                #     precs = [
                #         utils.validate(
                #             model, test_datasets[i], test_size=test_size,
                #             cuda=cuda, verbose=False,
                #         ) if i+1 <= task else 0 for i in
                #         range(len(train_datasets))
                #     ]
                #     title = (
                #         'precision (consolidated)' if consolidate else
                #         'precision'
                #     )
                #     visual.visualize_scalars(
                #         vis, precs, names, title,
                #         iteration
                #     )

                # # Send losses to the visdom server.
                # if iteration % loss_log_interval == 0:
                #     title = 'loss (consolidated)' if consolidate else 'loss'
                #     visual.visualize_scalars(
                #         vis,
                #         [loss, ce_loss, ewc_loss],
                #         ['total', 'cross entropy', 'ewc'],
                #         title, iteration
                #     )

        # for name, param in model.named_parameters():
        #     if param.requires_grad: print(name, param)

        if consolidate and task < len(train_datasets):
            # estimate the fisher information of the parameters and consolidate
            # them in the network.
            print(
                '=> Estimating diagonals of the fisher information matrix...',
                flush=True, end='',
            )
            model.consolidate(model.estimate_fisher(
                train_dataset, fisher_estimation_sample_size
            ))
            print(' Done!')
