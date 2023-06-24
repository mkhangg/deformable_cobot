import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log/train.log"),
        logging.StreamHandler()
    ]
)

logging.info('Test')
