import sys


def error_message_detail(error, error_detail: sys):
    """
    Captures details about the error, including the file name, line number, and error message.
    """
    # Get the traceback object from the error detail
    _, _, exc_tb = error_detail.exc_info()
    # Extract the file name and line number where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Format the error message
    error_message = "Error occurred in Python script name [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    """
    A custom exception class that provides detailed error messages,
    including the script name, line number, and error details.
    """

    def __init__(self, error_message, error_detail: sys):
        # Call the base class constructor
        super().__init__(error_message)
        # Generate the detailed error message
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        """
        Returns the detailed error message when the exception is converted to a string.
        """
        return self.error_message
