import numpy as np

from gym_rl_mpc.envs.base_turbine_env import BaseTurbineEnv
from gym_rl_mpc.objects.turbine import Turbine


class ConstantWind(BaseTurbineEnv):
    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

    def generate_environment(self):
        """
        Generates environment with a turbine and a random wind speed between min and max wind speed in config
        """
        self.wind_speed = (self.max_wind_speed - self.min_wind_speed) * self.rand_num_gen.rand() + self.min_wind_speed
        self.turbine = Turbine(self.wind_speed, self.step_size)

class ConstantWindLevel1(ConstantWind):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_wind_speed = 5
        self.min_wind_speed = 13

class ConstantWindLevel2(ConstantWind):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_wind_speed = 13
        self.min_wind_speed = 17

class BaseVariableWind(BaseTurbineEnv):
    """
    Subclasses must set variables: max_wind_amplitude, max_wind_speed, min_wind_speed, wind_noise
    If wind_noise is True, wind_noise_std must also be set.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def step(self, action):
        t = self.t_step * self.step_size

        wind_speed = self.wind_amplitude * np.sin(
            (1 / self.wind_period) * 2 * np.pi * t + self.wind_phase_shift) + self.wind_mean
        if self.wind_noise:
            wind_speed += np.random.normal(0, self.wind_noise_std)
        self.wind_speed = np.clip(wind_speed, self.min_wind_speed, self.max_wind_speed)

        return super().step(action)

    def generate_environment(self):
        """
        Generates environment with a turbine and a random wind speed between min and max wind speed in config
        """
        self.wind_amplitude = min((self.max_wind_speed - self.min_wind_speed) / 2,
                                  self.max_wind_amplitude) * self.rand_num_gen.rand()
        self.wind_mean = (self.max_wind_speed - self.min_wind_speed - 2 * self.wind_amplitude) * self.rand_num_gen.rand() + self.min_wind_speed + self.wind_amplitude
        self.wind_phase_shift = 2 * np.pi * self.rand_num_gen.rand()

        self.wind_speed = self.wind_amplitude * np.sin(self.wind_phase_shift) + self.wind_mean
        self.prev_wind_speed = self.wind_speed
        self.turbine = Turbine(self.wind_speed, self.step_size)


class VariableWindLevel0(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 1
        self.max_wind_speed = 17
        self.min_wind_speed = 13
        self.wind_noise = False
        super().__init__(*args, **kwargs)

class VariableWindLevel1(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 1
        self.max_wind_speed = 20
        self.min_wind_speed = 10
        self.wind_noise = False
        super().__init__(*args, **kwargs)


class VariableWindLevel2(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 1
        self.max_wind_speed = 25
        self.min_wind_speed = 5
        self.wind_noise = False
        super().__init__(*args, **kwargs)


class VariableWindLevel3(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 20
        self.min_wind_speed = 10
        self.wind_noise = False
        super().__init__(*args, **kwargs)

class VariableWindLevel4(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 25
        self.min_wind_speed = 10
        self.wind_noise = False
        super().__init__(*args, **kwargs)

class VariableWindLevel5(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 25
        self.min_wind_speed = 5
        self.wind_noise = False
        super().__init__(*args, **kwargs)

class VariableWindLevel6(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 25
        self.min_wind_speed = 10
        self.wind_noise = True
        self.wind_noise_std = 0.1
        super().__init__(*args, **kwargs)

class VariableWindLevel7(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 25
        self.min_wind_speed = 10
        self.wind_noise = True
        self.wind_noise_std = 0.2
        super().__init__(*args, **kwargs)

class VariableWindLevel8(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 25
        self.min_wind_speed = 10
        self.wind_noise = True
        self.wind_noise_std = 0.3
        super().__init__(*args, **kwargs)

class VariableWindLevel9(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 11
        self.min_wind_speed = 5
        self.wind_noise = False
        super().__init__(*args, **kwargs)
        
class VariableWindLevel10(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 25
        self.min_wind_speed = 15
        self.wind_noise = False
        super().__init__(*args, **kwargs)

class VariableWindPSFtest(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 25
        self.min_wind_speed = 13
        self.wind_noise = False
        super().__init__(*args, **kwargs)

class VariableWindPSFtestManual(BaseVariableWind):
    def __init__(self, *args, **kwargs) -> None:
        self.max_wind_amplitude = 3
        self.max_wind_speed = 25
        self.min_wind_speed = 13
        self.wind_noise = False
        super().__init__(*args, **kwargs)

    def generate_environment(self):
        """
        Generates environment with a turbine and a random wind speed between min and max wind speed in config
        """
        self.wind_amplitude = 3
        self.wind_phase_shift = 0

        self.wind_speed = self.wind_amplitude * np.sin(self.wind_phase_shift) + self.wind_mean
        self.prev_wind_speed = self.wind_speed
        self.turbine = Turbine(self.wind_speed, self.step_size)

class CrazyAgent(VariableWindLevel4):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def step(self, action):
        action = self.action_space_increase * self.action_space.sample()

        return super().step(action)
