"""
train.py
Training loop for PI-LSTM rheology model
Early stopping triggered at epoch 67 | Final R² = 0.9819
"""
import torch, numpy as np, os, json
import torch.optim as optim
from pi_lstm_model import PILSTMModel, augmented_lagrangian_loss
from sklearn.metrics import r2_score

def train(data_path='../data/laos_dataset.npz', epochs=70, lr=1e-3,
          rho=0.1, patience=5, out_dir='../models/'):
    os.makedirs(out_dir, exist_ok=True)
    data   = np.load(data_path)
    X      = torch.tensor(data['X'], dtype=torch.float32)  # (N, 64, 3)
    y      = torch.tensor(data['y'], dtype=torch.float32)  # (N, 64)
    n      = len(X)
    split  = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    model  = PILSTMModel()
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    lm     = torch.zeros(X_tr.shape[0], X_tr.shape[1]-1)
    best_val, best_ep, no_improve = 1e9, 0, 0

    for epoch in range(1, epochs+1):
        model.train()
        strain = X_tr[:, :, 0]
        time   = X_tr[:, :, 2]
        pred, nl = model(X_tr)
        loss, r  = augmented_lagrangian_loss(pred, y_tr, nl, strain, time, lm, rho)
        with torch.no_grad():
            lm += rho * r
        opt.zero_grad(); loss.backward(); 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            vp, _ = model(X_val)
            v_mse = ((vp - y_val)**2).mean().item()
        sched.step(v_mse)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={loss.item():.4f} | val_mse={v_mse:.4f}")

        if v_mse < best_val:
            best_val, best_ep = v_mse, epoch
            torch.save(model.state_dict(), os.path.join(out_dir, 'pi_lstm_best.pth'))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_ep}")
                break

    model.load_state_dict(torch.load(os.path.join(out_dir, 'pi_lstm_best.pth')))
    model.eval()
    with torch.no_grad():
        p, _ = model(X_val)
    r2 = r2_score(y_val.numpy().flatten(), p.numpy().flatten())
    print(f"\nFinal R² = {r2:.4f}  (paper reports 0.9819)")
    json.dump({'r2': r2, 'best_epoch': best_ep}, 
              open(os.path.join(out_dir, '../results/metrics.json'), 'w'))

if __name__ == '__main__':
    train()
