import argparse
from train_val import *
from sklearn.model_selection import KFold
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int, required=False,
                         help='..')
    parser.add_argument('--epochs', default=2000, type=int, required=False,
                         help='..')
    parser.add_argument('--lr', default=0.001, type=float, required=False,
                         help='..')
    parser.add_argument('--dataset', default='davis', type=str, required=False,
                        help='..')
    parser.add_argument('--save_log_path', default='default', type=str, required=False,
                         help='..')
    args = parser.parse_args()
    model = GINNet()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.0005, last_epoch=-1)


    train_dataset, valid_dataset = get_train_valid(dataset=args.dataset , fold = 0)
    test_dataset = get_test(dataset=args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate)
    train_eval(model, optimizer,scheduler, train_loader, val_loader, test_loader, args.epochs, log_path=args.save_log_path)


if __name__ == '__main__':
    main()
