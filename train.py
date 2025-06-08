import pytorch_lightning as pl
from lightning_module import PolypDetectionLightningModule
from data_module import PolypDataModule
import argparse

def main(args):
    # Initialize data module
    data_module = PolypDataModule(
        data_dir='.',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = PolypDetectionLightningModule(learning_rate=args.learning_rate)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',  # Automatically choose GPU if available
        devices=1,
        precision=16 if args.use_amp else 32,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath='checkpoints',
                filename='polyp-detection-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    
    args = parser.parse_args()
    main(args) 