import logging
formatter = logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', filemode='w')
# logging.basicConfig(format='[%(asctime)s] [%(filename)s:%(lineno)d] - %(message)s', 
# 					filemode='w') 

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger