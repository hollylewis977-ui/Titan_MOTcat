"""
Trainer functions for Hallmarks dataset with variable number of omic pathways.
No validation during training - only evaluate on test set at the final epoch.
"""
import numpy as np
import torch
import os

from sksurv.metrics import concordance_index_censored


def train_loop_survival_hallmarks(epoch, model, loader, optimizer, n_classes, writer=None,
                                   loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, args=None):
    """Training loop for Hallmarks dataset with variable omic count"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    # FiLM monitoring
    all_gamma = []
    all_beta = []

    for batch_idx, batch_data in enumerate(loader):
        # Dynamic unpacking: last 4 items are label, event_time, c, case_id
        data_WSI = batch_data[0].to(device)
        label = batch_data[-4].to(device)
        event_time = batch_data[-3]
        c = batch_data[-2].to(device)
        # case_id = batch_data[-1]  # Not needed during training

        # All items in between are omic features
        n_omics = len(batch_data) - 5  # WSI + omics + label + event_time + c + case_id
        omics = [batch_data[i].type(torch.FloatTensor).to(device) for i in range(1, n_omics + 1)]

        # Build model input kwargs
        kwargs = {'x_path': data_WSI}
        for i, omic in enumerate(omics):
            kwargs[f'x_omic{i+1}'] = omic

        hazards, S, Y_hat, A = model(**kwargs)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        # Collect FiLM parameters for monitoring
        if 'film_gamma' in A and 'film_beta' in A:
            all_gamma.append(A['film_gamma'].detach().cpu())
            all_beta.append(A['film_beta'].detach().cpu())

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
                batch_idx, loss_value, label.item(), float(event_time), float(risk))
            if args is not None and hasattr(args, 'writer_dir'):
                with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                    f.write(train_batch_str + '\n')
            print(train_batch_str)

        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    # Calculate loss and c-index for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index_train = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # FiLM monitoring: print gamma/beta statistics
    if len(all_gamma) > 0:
        gamma = torch.cat(all_gamma)
        beta = torch.cat(all_beta)
        film_str = 'Epoch {}: FiLM gamma mean={:.4f}, std={:.4f} | beta mean={:.4f}, std={:.4f}'.format(
            epoch, gamma.mean().item(), gamma.std().item(), beta.mean().item(), beta.std().item())
        print(film_str)
        if args is not None and hasattr(args, 'writer_dir'):
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(film_str + '\n')

    train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index_train)
    print(train_epoch_str)
    if args is not None and hasattr(args, 'writer_dir'):
        with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write(train_epoch_str + '\n')

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)

    return train_loss, c_index_train


def validate_survival_hallmarks(cur, epoch, model, loader, n_classes, loss_fn=None,
                                 reg_fn=None, lambda_reg=0., results_dir=None, args=None,
                                 train_events=None, train_times=None, bins=None):
    """Validation/Test loop for Hallmarks dataset with variable omic count"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_surv_probs = []

    patient_results = {}

    for batch_idx, batch_data in enumerate(loader):
        # Dynamic unpacking: last 4 items are label, event_time, c, case_id
        data_WSI = batch_data[0].to(device)
        label = batch_data[-4].to(device)
        event_time = batch_data[-3]
        c = batch_data[-2].to(device)
        case_id = batch_data[-1]  # String for patient identification

        n_omics = len(batch_data) - 5  # WSI + omics + label + event_time + c + case_id
        omics = [batch_data[i].type(torch.FloatTensor).to(device) for i in range(1, n_omics + 1)]

        kwargs = {'x_path': data_WSI}
        for i, omic in enumerate(omics):
            kwargs[f'x_omic{i+1}'] = omic

        with torch.no_grad():
            hazards, S, Y_hat, A = model(**kwargs)

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        all_surv_probs.append(S.cpu().numpy())

        # Use case_id as key for per-patient tracking (enables KM analysis, etc.)
        patient_results[case_id] = {
            'risk': risk.item() if hasattr(risk, 'item') else float(risk),
            'disc_label': label.item(),
            'survival': event_time.item() if hasattr(event_time, 'item') else float(event_time),
            'censorship': c.item(),
            'case_id': case_id  # Also store in value for convenience
        }

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # Compute IBS
    ibs = float('nan')
    if bins is not None and train_events is not None and train_times is not None:
        from utils.metrics import compute_ibs
        all_surv_probs = np.concatenate(all_surv_probs, axis=0)
        test_events = (1 - all_censorships).astype(bool)
        ibs = compute_ibs(train_events, train_times,
                          test_events, all_event_times,
                          all_surv_probs, bins)

    val_epoch_str = "test c-index: {:.4f}, IBS: {:.4f}".format(c_index, ibs)
    print(val_epoch_str)
    if args is not None and hasattr(args, 'writer_dir'):
        with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write(val_epoch_str + '\n')

    return patient_results, c_index, ibs


def train_loop_survival_hallmarks_mb(epoch, bs_micro, model, loader, optimizer, n_classes,
                                      writer=None, loss_fn=None, reg_fn=None, lambda_reg=0.,
                                      gc=32, args=None):
    """Training loop with micro-batching for large WSI bags"""
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    # FiLM monitoring
    all_gamma = []
    all_beta = []

    for batch_idx, batch_data in enumerate(loader):
        # Dynamic unpacking: last 4 items are label, event_time, c, case_id
        data_WSI = batch_data[0]  # Keep on CPU for chunking
        label = batch_data[-4].to(device)
        event_time = batch_data[-3]
        c = batch_data[-2].to(device)
        # case_id = batch_data[-1]  # Not needed during training

        n_omics = len(batch_data) - 5  # WSI + omics + label + event_time + c + case_id
        omics = [batch_data[i].type(torch.FloatTensor).to(device) for i in range(1, n_omics + 1)]

        loss = 0.
        all_risk = 0.
        cnt = 0

        # Split WSI into micro-batches
        index_chunk_list = split_chunk_list(data_WSI, bs_micro)
        for tindex in index_chunk_list:
            wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex)).to(device)

            kwargs = {'x_path': wsi_mb}
            for i, omic in enumerate(omics):
                kwargs[f'x_omic{i+1}'] = omic

            hazards, S, Y_hat, A = model(**kwargs)
            loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            loss += loss_micro
            all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
            cnt += 1

            # Collect FiLM parameters from last micro-batch
            if 'film_gamma' in A and 'film_beta' in A:
                last_gamma = A['film_gamma'].detach().cpu()
                last_beta = A['film_beta'].detach().cpu()

        # Store FiLM from last micro-batch of this sample
        if 'last_gamma' in locals():
            all_gamma.append(last_gamma)
            all_beta.append(last_beta)

        loss = loss / cnt
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = all_risk / cnt
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 50 == 0:
            train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
                batch_idx, loss_value, label.item(), float(event_time), float(risk))
            if args is not None and hasattr(args, 'writer_dir'):
                with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                    f.write(train_batch_str + '\n')
            print(train_batch_str)

        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index_train = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # FiLM monitoring: print gamma/beta statistics
    if len(all_gamma) > 0:
        gamma = torch.cat(all_gamma)
        beta = torch.cat(all_beta)
        film_str = 'Epoch {}: FiLM gamma mean={:.4f}, std={:.4f} | beta mean={:.4f}, std={:.4f}'.format(
            epoch, gamma.mean().item(), gamma.std().item(), beta.mean().item(), beta.std().item())
        print(film_str)
        if args is not None and hasattr(args, 'writer_dir'):
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(film_str + '\n')

    train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index_train)
    print(train_epoch_str)
    if args is not None and hasattr(args, 'writer_dir'):
        with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write(train_epoch_str + '\n')

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)

    return train_loss, c_index_train


def validate_survival_hallmarks_mb(cur, epoch, bs_micro, model, loader, n_classes,
                                    loss_fn=None, reg_fn=None, lambda_reg=0.,
                                    results_dir=None, args=None,
                                    train_events=None, train_times=None, bins=None):
    """Validation with micro-batching"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_surv_probs = []

    patient_results = {}

    for batch_idx, batch_data in enumerate(loader):
        # Dynamic unpacking: last 4 items are label, event_time, c, case_id
        data_WSI = batch_data[0]
        label = batch_data[-4].to(device)
        event_time = batch_data[-3]
        c = batch_data[-2].to(device)
        case_id = batch_data[-1]  # String for patient identification

        n_omics = len(batch_data) - 5  # WSI + omics + label + event_time + c + case_id
        omics = [batch_data[i].type(torch.FloatTensor).to(device) for i in range(1, n_omics + 1)]

        loss = 0.
        all_risk = 0.
        surv_acc = None
        cnt = 0

        with torch.no_grad():
            index_chunk_list = split_chunk_list(data_WSI, bs_micro)
            for tindex in index_chunk_list:
                wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex)).to(device)

                kwargs = {'x_path': wsi_mb}
                for i, omic in enumerate(omics):
                    kwargs[f'x_omic{i+1}'] = omic

                hazards, S, Y_hat, A = model(**kwargs)
                loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
                loss += loss_micro
                all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
                if surv_acc is None:
                    surv_acc = S.detach().cpu().numpy()
                else:
                    surv_acc = surv_acc + S.detach().cpu().numpy()
                cnt += 1

        loss = loss / cnt
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = all_risk / cnt
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        all_surv_probs.append(surv_acc / cnt)

        # Use case_id as key for per-patient tracking (enables KM analysis, etc.)
        patient_results[case_id] = {
            'risk': risk,
            'disc_label': label.item(),
            'survival': event_time.item() if hasattr(event_time, 'item') else float(event_time),
            'censorship': c.item(),
            'case_id': case_id  # Also store in value for convenience
        }

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # Compute IBS
    ibs = float('nan')
    if bins is not None and train_events is not None and train_times is not None:
        from utils.metrics import compute_ibs
        all_surv_probs = np.concatenate(all_surv_probs, axis=0)
        test_events = (1 - all_censorships).astype(bool)
        ibs = compute_ibs(train_events, train_times,
                          test_events, all_event_times,
                          all_surv_probs, bins)

    val_epoch_str = "test c-index: {:.4f}, IBS: {:.4f}".format(c_index, ibs)
    print(val_epoch_str)
    if args is not None and hasattr(args, 'writer_dir'):
        with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write(val_epoch_str + '\n')

    return patient_results, c_index, ibs


def split_chunk_list(data, batch_size):
    """Split data indices into chunks for micro-batching"""
    import random
    numGroup = data.shape[0] // batch_size + 1
    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    return index_chunk_list
