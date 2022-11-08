import argparse
from train_val import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int, required=False,
                         help='..')
    parser.add_argument('--epochs', default=2000, type=int, required=False,
                         help='..')
    parser.add_argument('--lr', default=0.001, type=float, required=False,
                         help='..')
    parser.add_argument('--load_model_path', default=None, type=str, required=False,
                        help='..')
    parser.add_argument('--save_log_path', default='default', type=str, required=False,
                         help='..')
    args = parser.parse_args()
    model = GINNet()

    if args.load_model_path is not None:
        print(args.load_model_path)
        save_model = torch.load(args.load_model_path)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.0005, last_epoch=-1)



    dataset = get_finetune( dataset='covid_finetune')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate)
    train_eval(model, optimizer,scheduler, dataloader, dataloader, None, args.epochs, log_path=args.save_log_path)

if __name__ == '__main__':
    main()
