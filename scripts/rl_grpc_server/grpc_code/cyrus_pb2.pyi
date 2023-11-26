from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Vec2D(_message.Message):
    __slots__ = ["X", "Y"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    X: float
    Y: float
    def __init__(self, X: _Optional[float] = ..., Y: _Optional[float] = ...) -> None: ...

class Ang2D(_message.Message):
    __slots__ = ["Angle"]
    ANGLE_FIELD_NUMBER: _ClassVar[int]
    Angle: float
    def __init__(self, Angle: _Optional[float] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ["Position", "Body", "Cycle"]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    CYCLE_FIELD_NUMBER: _ClassVar[int]
    Position: Vec2D
    Body: Ang2D
    Cycle: int
    def __init__(self, Position: _Optional[_Union[Vec2D, _Mapping]] = ..., Body: _Optional[_Union[Ang2D, _Mapping]] = ..., Cycle: _Optional[int] = ...) -> None: ...

class ActionDash(_message.Message):
    __slots__ = ["Power", "Dir"]
    POWER_FIELD_NUMBER: _ClassVar[int]
    DIR_FIELD_NUMBER: _ClassVar[int]
    Power: float
    Dir: Ang2D
    def __init__(self, Power: _Optional[float] = ..., Dir: _Optional[_Union[Ang2D, _Mapping]] = ...) -> None: ...

class ActionTurn(_message.Message):
    __slots__ = ["Dir"]
    DIR_FIELD_NUMBER: _ClassVar[int]
    Dir: Ang2D
    def __init__(self, Dir: _Optional[_Union[Ang2D, _Mapping]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ["Dash", "Turn"]
    DASH_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    Dash: ActionDash
    Turn: ActionTurn
    def __init__(self, Dash: _Optional[_Union[ActionDash, _Mapping]] = ..., Turn: _Optional[_Union[ActionTurn, _Mapping]] = ...) -> None: ...

class Reward(_message.Message):
    __slots__ = ["Value", "Cycle", "Unum", "Done"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    CYCLE_FIELD_NUMBER: _ClassVar[int]
    UNUM_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    Value: float
    Cycle: int
    Unum: int
    Done: bool
    def __init__(self, Value: _Optional[float] = ..., Cycle: _Optional[int] = ..., Unum: _Optional[int] = ..., Done: bool = ...) -> None: ...

class StartEpisode(_message.Message):
    __slots__ = ["Cycle"]
    CYCLE_FIELD_NUMBER: _ClassVar[int]
    Cycle: int
    def __init__(self, Cycle: _Optional[int] = ...) -> None: ...

class TrainerRequest(_message.Message):
    __slots__ = ["StartEpisode", "Reward"]
    STARTEPISODE_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    StartEpisode: StartEpisode
    Reward: Reward
    def __init__(self, StartEpisode: _Optional[_Union[StartEpisode, _Mapping]] = ..., Reward: _Optional[_Union[Reward, _Mapping]] = ...) -> None: ...

class OK(_message.Message):
    __slots__ = ["OK"]
    OK_FIELD_NUMBER: _ClassVar[int]
    OK: bool
    def __init__(self, OK: bool = ...) -> None: ...
