import sys
from logger import logging


# Configure logging
logging.basicConfig(
    filename="error.log",  # Log to a file
    level=logging.ERROR,   # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def error_message_details(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomeException(Exception):
    def __init__(self, error_message, error_details: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)
    
    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 10
        b = 0
        c = a / b
    except Exception as e:
        # Log the custom error message
        logging.error(str(CustomeException(e, sys)))
        # Gracefully terminate with the error message
        print(str(CustomeException(e, sys)))
