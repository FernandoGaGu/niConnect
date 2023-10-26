import configparser
import os
from datetime import datetime

COLORS = {
    'purple': '\033[95m',
    'cyan': '\033[96m',
    'blue': '\033[94m',
    'green': '\033[92m',
    'red': '\033[91m',
    'yellow': '\033[93m',
    'END': '\033[0m'
}

# Entry point associated with the configuration directories generated by the script
# 's2_estimate_connectivity.py'
CONFIGURATION_ENTER_POINT = '.niconnectConfigEnterPoint'

# Entry point associated with the configuration directories generated by the script
# 's3_optimize_networks.py'
OPTIMIZATION_ENTER_POINT = '.niconnectOptimEnterPoint'


class INIReader(object):
    @classmethod
    def parseFile(
        cls,
        file: str,
        required_sections: list = None
    ):
        assert isinstance(file, str), 'Input variable file must be a string'
        assert os.path.exists(file), 'File %s not found' % file
        assert not os.path.isdir(file), '%s is not a directory' % file

        config = configparser.ConfigParser()
        config.read(file)

        if required_sections is not None:
            assert isinstance(required_sections, list)
            for section in required_sections:
                if section not in config.sections():
                    raise TypeError('Section "%s" not found in configuration file "%s"' % (section, file))

        return config


def pprint(message: str, color: str):
    """
    Displays by console the specified message formatted in the specified colour if possible.


    Parameters
    ----------
    message: str
        Message to be displayed on the screen.

    color: str, default=None
        Colour of the message to be displayed on the screen. To see available colors use:
        help(niconnect.system.COLORS).
    """
    if color in COLORS:
        print(COLORS[color], end='')
        print(message, end='')
        print(COLORS['END'])
    else:
        print(message)


def defaultName() -> str:
    """
    Function that returns a default file name based on the schema: MONTH-DAY-YEAR-HOUR-MIN-SEC.

    Returns
    -------
    :str
        Default name.
    """
    return datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


def createEnterPoint(enter_point: str, output_dir: str):
    """ Function that creates a hidden file that acts as an entry point to identify specific
    directories. """
    assert os.path.exists(output_dir), '%s not found.' % output_dir
    assert os.path.isdir(output_dir), '%s is not a directory' % output_dir

    with open('%s/%s' % (output_dir, enter_point), 'w') as file:
        file.write(defaultName())


if __name__ == '__main__':
    config = INIReader.parseFile(
        os.path.join('..', 'config', 's1_extract_suvr.ini'),
        ['IN.DATA', 'OUT.DATA']
    )

    print(list(config['IN.DATA'].items()))

