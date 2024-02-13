import time

from datetime import datetime


def get_hex_time(ms=False):
    """
    Description:
        get the current time in the format "DD/MM/YY HH:MM:SS" and convert it to a hexadecimal string
    """
    if ms:
        # Get current time with microseconds
        current_time = datetime.now().strftime("%d/%m/%y %H:%M:%S.%f")

        # Convert the time string to a datetime object
        dt_object = datetime.strptime(current_time, "%d/%m/%y %H:%M:%S.%f")

        # Convert the datetime object to a Unix timestamp with microseconds
        unix_time_with_microseconds = dt_object.timestamp()

        # Convert the Unix timestamp to a hexadecimal string, slicing off the '0x' and the 'L' at the end if it exists
        hex_time = hex(int(unix_time_with_microseconds * 1e6))[2:]

    else:
        current_time = time.strftime("%d/%m/%y %H:%M:%S", time.localtime())
        # convert the timestamp string to a Unix timestamp
        unix_time = int(time.mktime(time.strptime(current_time, "%d/%m/%y %H:%M:%S")))

        # convert the Unix timestamp to a hexadecimal string
        hex_time = hex(unix_time)[2:]

    return hex_time


def hex_to_time(hex_time, ms=False):
    """
    input:
        hex_time: str
    description:
        convert a hexadecimal string to a timestamp string in the format "DD/MM/YY HH:MM:SS"
    """
    # convert the hexadecimal string to a Unix timestamp
    if ms:
        # Convert the hexadecimal string to a Unix timestamp including microseconds
        unix_time_with_microseconds = (
            int(hex_time, 16) / 1e6
        )  # Divide by 1e6 to convert microseconds to seconds

        # Convert the Unix timestamp to a datetime object
        dt_object = datetime.fromtimestamp(unix_time_with_microseconds)

        # Format the datetime object to a string including microseconds
        time_str = dt_object.strftime("%d/%m/%y %H:%M:%S.%f")

    else:
        unix_time = int(hex_time, 16)

        # convert the Unix timestamp to a timestamp string in the format "DD/MM/YY HH:MM:SS"
        time_str = time.strftime("%d/%m/%y %H:%M:%S", time.localtime(unix_time))

    return time_str


if __name__ == "__main__":
    a = get_hex_time(ms=True)
    print(a)
    print(hex_to_time(a, ms=True))
