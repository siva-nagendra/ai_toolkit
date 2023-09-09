import argparse
import os
from helpers.train_custom_dataset import run as train_run
import subprocess


def main():
    """
    A CLI tool to load a vector database or train a model with a specified dataset.
    Utilizes argparse to handle command line inputs and directs the flow of operation 
    based on the command and arguments given.
    
    Raises:
        ValueError: If the necessary environment variables or command options are not provided.
    """
    parser = argparse.ArgumentParser(description='USD AI Command Line Interface')
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Load Command
    load_parser = subparsers.add_parser('load', help='Load the vector database')
    load_parser.add_argument('--db-path', type=str, default=os.getenv('DB_FAISS_PATH'), help='Path to vector database')
    
    # Train Command
    train_parser = subparsers.add_parser('train', help='Train the model with specified dataset')
    train_parser.add_argument('--dataset-path', type=str, default=os.getenv('DATASET_PATH'), help='Path to the dataset')
    train_parser.add_argument('--db-path', type=str, default=os.getenv('DB_FAISS_PATH'), help='Path to vector database')
    
    args = parser.parse_args()
    
    if args.command == 'load':
        if args.db_path:
            subprocess.run(["chainlit", "run", "model.py"])
        else:
            raise ValueError("DB_FAISS_PATH not set. Please set the DB_FAISS_PATH environment variable or use the --db-path option.")
    
    elif args.command == 'train':
        if args.dataset_path and args.db_path:
            train_run(args.dataset_path, args.db_path)
        else:
            raise ValueError("Please ensure both DATASET_PATH and DB_FAISS_PATH are set or use the --dataset-path and --db-path options.")


if __name__ == "__main__":
    main()
