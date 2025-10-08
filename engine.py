

from typing import Dict, List
import torch.nn as nn
import os



from typing import Tuple
from tqdm import tqdm
import torch

from sklearn.metrics import f1_score, precision_score, recall_score


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler,
               epoch_num: int,
               device: torch.device) -> Tuple[float, float, float, float]:
    model.train()
    batch_losses = []
    batch_f1s = []
    batch_precisions = []
    batch_recalls = []

    prog_bar = tqdm(dataloader,
                    desc=f"Train Epoch {epoch_num + 1}",
                    unit="batch",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    for batch, (X, y) in enumerate(prog_bar):
        X, y = X.to(device), y.to(device)

        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        optimizer.zero_grad()
        y_pred,_ = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        scheduler.step()


        preds = y_pred.argmax(dim=1)

        batch_preds = preds.cpu().numpy()
        batch_targets = y.cpu().numpy()
        batch_loss = loss.item()



        batch_f1 = f1_score(batch_targets, batch_preds, average='weighted', zero_division=0)
        batch_precision = precision_score(batch_targets, batch_preds, average='weighted', zero_division=0)
        batch_recall = recall_score(batch_targets, batch_preds, average='weighted', zero_division=0)

        batch_losses.append(batch_loss)
        batch_f1s.append(batch_f1)
        batch_precisions.append(batch_precision)
        batch_recalls.append(batch_recall)




        prog_bar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'f1': f'{batch_f1:.4f}',
            'precision': f'{batch_precision:.4f}',
            'recall': f'{batch_recall:.4f}',
            "current_lr":f"{current_lr:.6f}"
        }, refresh=True)


    final_loss = sum(batch_losses) / len(batch_losses)

    final_f1 = sum(batch_f1s) / len(batch_f1s)
    final_precision = sum(batch_precisions) / len(batch_precisions)
    final_recall = sum(batch_recalls) / len(batch_recalls)

    return final_loss,final_f1, final_precision, final_recall


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              epoch_num: int,
              device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()


    batch_losses = []
    batch_f1s = []
    batch_precisions = []
    batch_recalls = []

    prog_bar = tqdm(dataloader,
                    desc=f"Test Epoch {epoch_num + 1}",
                    unit="batch",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    with torch.inference_mode():
        for batch, (X, y) in enumerate(prog_bar):
            X, y = X.to(device), y.to(device)

            y_pred,_ = model(X)
            loss = loss_fn(y_pred, y)

            preds = y_pred.argmax(dim=1)

            batch_preds = preds.cpu().numpy()
            batch_targets = y.cpu().numpy()
            batch_loss = loss.item()

            batch_f1 = f1_score(batch_targets, batch_preds, average='weighted', zero_division=0)
            batch_precision = precision_score(batch_targets, batch_preds, average='weighted', zero_division=0)
            batch_recall = recall_score(batch_targets, batch_preds, average='weighted', zero_division=0)

            batch_losses.append(batch_loss)

            batch_f1s.append(batch_f1)
            batch_precisions.append(batch_precision)
            batch_recalls.append(batch_recall)



            prog_bar.set_postfix({
                'loss': f'{ batch_loss:.4f}',
                'f1': f'{batch_f1:.4f}',
                'precision': f'{batch_precision:.4f}',
                'recall': f'{batch_recall:.4f}'
            }, refresh=True)




    final_loss = sum(batch_losses) / len(batch_losses)

    final_f1 = sum(batch_f1s) / len(batch_f1s)
    final_precision = sum(batch_precisions) / len(batch_precisions)
    final_recall = sum(batch_recalls) / len(batch_recalls)



    return final_loss, final_f1, final_precision, final_recall


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          scheduler,
          device: torch.device) -> Dict[str, List]:
    results = {
        "train_loss": [], "train_f1": [], "train_precision": [], "train_recall": [],
        "test_loss": [], "test_f1": [], "test_precision": [], "test_recall": []
    }

    model.to(device)
    best_acc = 0.0
    best_model_path = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)

        train_loss,train_f1, train_prec, train_rec = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            device=device)

        test_loss, test_f1, test_prec, test_rec = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            epoch_num=epoch,
            device=device)

        results["train_loss"].append(train_loss)

        results["train_f1"].append(train_f1)
        results["train_precision"].append(train_prec)
        results["train_recall"].append(train_rec)

        results["test_loss"].append(test_loss)

        results["test_f1"].append(test_f1)
        results["test_precision"].append(test_prec)
        results["test_recall"].append(test_rec)



        print(f"\nEpoch {epoch + 1} Summary (Averages):")
        print(
            f"  Train - Loss: {train_loss:.4f}, "
            f"F1: {train_f1:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}"
        )
        print(
            f"  Test  - Loss: {test_loss:.4f}, "
            f"F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}"
        )

        checkpoint_path = os.path.join("checkpoints", f"epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": test_loss,
        }, checkpoint_path)

        print(f"Model saved at {checkpoint_path}")

        # Save best model (based on test accuracy)
        if test_f1 > best_acc:
            best_acc = test_f1
            best_model_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model updated! (Acc: {best_acc:.4f}) saved at {best_model_path}")

    print(f"\nTraining complete! Best model accuracy: {best_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")


    return results



def train_kd_step(student: torch.nn.Module,teacher: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler,
               epoch_num: int,
               device: torch.device) -> Tuple[float, float, float, float]:
    student.train()
    teacher.eval()


    batch_losses = []
    batch_f1s = []
    batch_precisions = []
    batch_recalls = []

    prog_bar = tqdm(dataloader,
                    desc=f"Train Epoch {epoch_num + 1}",
                    unit="batch",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    for batch, (X, y) in enumerate(prog_bar):
        X, y = X.to(device), y.to(device)

        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits,_ = teacher(X)


        student_logits,_ = student(X)


        loss = loss_fn(student_logits,teacher_logits ,y)
        loss.backward()
        optimizer.step()

        scheduler.step()


        preds = student_logits.argmax(dim=1)

        batch_preds = preds.cpu().numpy()
        batch_targets = y.cpu().numpy()
        batch_loss = loss.item()



        batch_f1 = f1_score(batch_targets, batch_preds, average='weighted', zero_division=0)
        batch_precision = precision_score(batch_targets, batch_preds, average='weighted', zero_division=0)
        batch_recall = recall_score(batch_targets, batch_preds, average='weighted', zero_division=0)

        batch_losses.append(batch_loss)
        batch_f1s.append(batch_f1)
        batch_precisions.append(batch_precision)
        batch_recalls.append(batch_recall)




        prog_bar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'f1': f'{batch_f1:.4f}',
            'precision': f'{batch_precision:.4f}',
            'recall': f'{batch_recall:.4f}',
            "current_lr":f"{current_lr:.6f}"
        }, refresh=True)


    final_loss = sum(batch_losses) / len(batch_losses)

    final_f1 = sum(batch_f1s) / len(batch_f1s)
    final_precision = sum(batch_precisions) / len(batch_precisions)
    final_recall = sum(batch_recalls) / len(batch_recalls)

    return final_loss,final_f1, final_precision, final_recall

def test_kd_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              epoch_num: int,
              device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()


    batch_losses = []
    batch_f1s = []
    batch_precisions = []
    batch_recalls = []

    prog_bar = tqdm(dataloader,
                    desc=f"Test Epoch {epoch_num + 1}",
                    unit="batch",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    with torch.inference_mode():
        for batch, (X, y) in enumerate(prog_bar):
            X, y = X.to(device), y.to(device)

            y_pred,_ = model(X)
            loss = loss_fn(y_pred, y)

            preds = y_pred.argmax(dim=1)

            batch_preds = preds.cpu().numpy()
            batch_targets = y.cpu().numpy()
            batch_loss = loss.item()

            batch_f1 = f1_score(batch_targets, batch_preds, average='weighted', zero_division=0)
            batch_precision = precision_score(batch_targets, batch_preds, average='weighted', zero_division=0)
            batch_recall = recall_score(batch_targets, batch_preds, average='weighted', zero_division=0)

            batch_losses.append(batch_loss)

            batch_f1s.append(batch_f1)
            batch_precisions.append(batch_precision)
            batch_recalls.append(batch_recall)



            prog_bar.set_postfix({
                'loss': f'{ batch_loss:.4f}',
                'f1': f'{batch_f1:.4f}',
                'precision': f'{batch_precision:.4f}',
                'recall': f'{batch_recall:.4f}'
            }, refresh=True)




    final_loss = sum(batch_losses) / len(batch_losses)

    final_f1 = sum(batch_f1s) / len(batch_f1s)
    final_precision = sum(batch_precisions) / len(batch_precisions)
    final_recall = sum(batch_recalls) / len(batch_recalls)



    return final_loss, final_f1, final_precision, final_recall


def train_kd(student: torch.nn.Module,teacher: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,kd: torch.nn.Module,
          epochs: int,
          scheduler,
          device: torch.device) -> Dict[str, List]:
    results = {
        "train_loss": [], "train_f1": [], "train_precision": [], "train_recall": [],
        "test_loss": [], "test_f1": [], "test_precision": [], "test_recall": []
    }

    student.to(device)
    teacher.to(device)

    best_acc = 0.0
    best_model_path = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)

        train_loss,train_f1, train_prec, train_rec = train_kd_step(
            student=student,
            teacher=teacher,
            dataloader=train_dataloader,
            loss_fn=kd,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            device=device)

        test_loss, test_f1, test_prec, test_rec = test_kd_step(
            model=student,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            epoch_num=epoch,
            device=device)

        results["train_loss"].append(train_loss)

        results["train_f1"].append(train_f1)
        results["train_precision"].append(train_prec)
        results["train_recall"].append(train_rec)

        results["test_loss"].append(test_loss)

        results["test_f1"].append(test_f1)
        results["test_precision"].append(test_prec)
        results["test_recall"].append(test_rec)



        print(f"\nEpoch {epoch + 1} Summary (Averages):")
        print(
            f"  Train - Loss: {train_loss:.4f}, "
            f"F1: {train_f1:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}"
        )
        print(
            f"  Test  - Loss: {test_loss:.4f}, "
            f"F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}"
        )

        checkpoint_path = os.path.join("checkpoints", f"epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": test_loss,
        }, checkpoint_path)

        print(f"Model saved at {checkpoint_path}")

        # Save best model (based on test accuracy)
        if test_f1 > best_acc:
            best_acc = test_f1
            best_model_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(student.state_dict(), best_model_path)
            print(f"✅ Best model updated! (Acc: {best_acc:.4f}) saved at {best_model_path}")

    print(f"\nTraining complete! Best model accuracy: {best_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")


    return results


def train_AT_step(student: torch.nn.Module,teacher: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler,
               epoch_num: int,
               device: torch.device) -> Tuple[float, float, float, float]:
    student.train()
    teacher.eval()


    batch_losses = []
    batch_f1s = []
    batch_precisions = []
    batch_recalls = []

    prog_bar = tqdm(dataloader,
                    desc=f"Train Epoch {epoch_num + 1}",
                    unit="batch",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    for batch, (X, y) in enumerate(prog_bar):
        X, y = X.to(device), y.to(device)

        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits,feats_t = teacher(X)


        student_logits,feats_s = student(X)


        loss = loss_fn(student_logits,feats_s ,feats_t ,y)
        loss.backward()
        optimizer.step()

        scheduler.step()


        preds = student_logits.argmax(dim=1)

        batch_preds = preds.cpu().numpy()
        batch_targets = y.cpu().numpy()
        batch_loss = loss.item()



        batch_f1 = f1_score(batch_targets, batch_preds, average='weighted', zero_division=0)
        batch_precision = precision_score(batch_targets, batch_preds, average='weighted', zero_division=0)
        batch_recall = recall_score(batch_targets, batch_preds, average='weighted', zero_division=0)

        batch_losses.append(batch_loss)
        batch_f1s.append(batch_f1)
        batch_precisions.append(batch_precision)
        batch_recalls.append(batch_recall)




        prog_bar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'f1': f'{batch_f1:.4f}',
            'precision': f'{batch_precision:.4f}',
            'recall': f'{batch_recall:.4f}',
            "current_lr":f"{current_lr:.6f}"
        }, refresh=True)


    final_loss = sum(batch_losses) / len(batch_losses)

    final_f1 = sum(batch_f1s) / len(batch_f1s)
    final_precision = sum(batch_precisions) / len(batch_precisions)
    final_recall = sum(batch_recalls) / len(batch_recalls)

    return final_loss,final_f1, final_precision, final_recall


def train_AT(student: torch.nn.Module,teacher: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,kd: torch.nn.Module,
          epochs: int,
          scheduler,
          device: torch.device) -> Dict[str, List]:
    results = {
        "train_loss": [], "train_f1": [], "train_precision": [], "train_recall": [],
        "test_loss": [], "test_f1": [], "test_precision": [], "test_recall": []
    }

    student.to(device)
    teacher.to(device)

    best_acc = 0.0
    best_model_path = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)

        train_loss,train_f1, train_prec, train_rec = train_AT_step(
            student=student,
            teacher=teacher,
            dataloader=train_dataloader,
            loss_fn=kd,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            device=device)

        test_loss, test_f1, test_prec, test_rec = test_kd_step(
            model=student,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            epoch_num=epoch,
            device=device)

        results["train_loss"].append(train_loss)

        results["train_f1"].append(train_f1)
        results["train_precision"].append(train_prec)
        results["train_recall"].append(train_rec)

        results["test_loss"].append(test_loss)

        results["test_f1"].append(test_f1)
        results["test_precision"].append(test_prec)
        results["test_recall"].append(test_rec)



        print(f"\nEpoch {epoch + 1} Summary (Averages):")
        print(
            f"  Train - Loss: {train_loss:.4f}, "
            f"F1: {train_f1:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}"
        )
        print(
            f"  Test  - Loss: {test_loss:.4f}, "
            f"F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}"
        )

        checkpoint_path = os.path.join("checkpoints", f"epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": test_loss,
        }, checkpoint_path)

        print(f"Model saved at {checkpoint_path}")

        # Save best model (based on test accuracy)
        if test_f1 > best_acc:
            best_acc = test_f1
            best_model_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(student.state_dict(), best_model_path)
            print(f"✅ Best model updated! (Acc: {best_acc:.4f}) saved at {best_model_path}")

    print(f"\nTraining complete! Best model accuracy: {best_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")


    return results
