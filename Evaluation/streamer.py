import sounddevice as sd
from singleton_decorator import singleton

@singleton
class Streamer:
    def __init__(self, chunk_size, sample_rate, channels=1):
        device = self.get_available_devices()
        if device is not None:
            device_indx, sr = self.choose_device(device)
        else:
            raise Exception('Please insert a device to continue...')

        self._stream = sd.InputStream(device=device_indx,
                                      channels=channels,
                                      samplerate=sample_rate,
                                      blocksize=chunk_size)
        self._stream.start()

    @property
    def stream(self):
        return self._stream

    @staticmethod
    def get_available_devices():
        devices, temp = [], []
        for indx, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append(indx)
                temp.append((indx, dev['name'], dev['default_samplerate']))

        if len(devices):
            for indx, name, sr in temp:
                print(f'Device {indx}. is {name} and default Sample Rate is {sr}')
            return devices
        else:
            return None

    @staticmethod
    def choose_device(input_devices):
        dev_indx = -2
        while dev_indx not in input_devices:
            print('Please type input device ID:')
            dev_indx = int(input())

        # Automatically select the default sample rate for the chosen device
        sr = sd.query_devices(dev_indx, 'input')['default_samplerate']
        return dev_indx, sr
