from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Vec2D(_message.Message):
    __slots__ = ("X", "Y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    X: float
    Y: float
    def __init__(self, X: _Optional[float] = ..., Y: _Optional[float] = ...) -> None: ...

class GoToPointStateMessage(_message.Message):
    __slots__ = ("Position", "Body", "BodyDiff")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    BODYDIFF_FIELD_NUMBER: _ClassVar[int]
    Position: Vec2D
    Body: float
    BodyDiff: float
    def __init__(self, Position: _Optional[_Union[Vec2D, _Mapping]] = ..., Body: _Optional[float] = ..., BodyDiff: _Optional[float] = ...) -> None: ...

class GoToBallStateMessage(_message.Message):
    __slots__ = ("Position", "Body", "BallPosition", "BodyDiff")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    BALLPOSITION_FIELD_NUMBER: _ClassVar[int]
    BODYDIFF_FIELD_NUMBER: _ClassVar[int]
    Position: Vec2D
    Body: float
    BallPosition: Vec2D
    BodyDiff: float
    def __init__(self, Position: _Optional[_Union[Vec2D, _Mapping]] = ..., Body: _Optional[float] = ..., BallPosition: _Optional[_Union[Vec2D, _Mapping]] = ..., BodyDiff: _Optional[float] = ...) -> None: ...

class RawListMessage(_message.Message):
    __slots__ = ("Value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    Value: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, Value: _Optional[_Iterable[float]] = ...) -> None: ...

class StateMessage(_message.Message):
    __slots__ = ("Cycle", "RawList", "GoToPointState", "GoToBallState")
    CYCLE_FIELD_NUMBER: _ClassVar[int]
    RAWLIST_FIELD_NUMBER: _ClassVar[int]
    GOTOPOINTSTATE_FIELD_NUMBER: _ClassVar[int]
    GOTOBALLSTATE_FIELD_NUMBER: _ClassVar[int]
    Cycle: int
    RawList: RawListMessage
    GoToPointState: GoToPointStateMessage
    GoToBallState: GoToBallStateMessage
    def __init__(self, Cycle: _Optional[int] = ..., RawList: _Optional[_Union[RawListMessage, _Mapping]] = ..., GoToPointState: _Optional[_Union[GoToPointStateMessage, _Mapping]] = ..., GoToBallState: _Optional[_Union[GoToBallStateMessage, _Mapping]] = ...) -> None: ...

class ActionDashMessage(_message.Message):
    __slots__ = ("Power", "Dir")
    POWER_FIELD_NUMBER: _ClassVar[int]
    DIR_FIELD_NUMBER: _ClassVar[int]
    Power: float
    Dir: float
    def __init__(self, Power: _Optional[float] = ..., Dir: _Optional[float] = ...) -> None: ...

class ActionTurnMessage(_message.Message):
    __slots__ = ("Dir",)
    DIR_FIELD_NUMBER: _ClassVar[int]
    Dir: float
    def __init__(self, Dir: _Optional[float] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("RawList", "Dash", "Turn")
    RAWLIST_FIELD_NUMBER: _ClassVar[int]
    DASH_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    RawList: RawListMessage
    Dash: ActionDashMessage
    Turn: ActionTurnMessage
    def __init__(self, RawList: _Optional[_Union[RawListMessage, _Mapping]] = ..., Dash: _Optional[_Union[ActionDashMessage, _Mapping]] = ..., Turn: _Optional[_Union[ActionTurnMessage, _Mapping]] = ...) -> None: ...

class TrainerRequest(_message.Message):
    __slots__ = ("Cycle", "Reward", "Unum", "Done", "Start", "Statud")
    CYCLE_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    UNUM_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    STATUD_FIELD_NUMBER: _ClassVar[int]
    Cycle: int
    Reward: float
    Unum: int
    Done: bool
    Start: bool
    Statud: int
    def __init__(self, Cycle: _Optional[int] = ..., Reward: _Optional[float] = ..., Unum: _Optional[int] = ..., Done: bool = ..., Start: bool = ..., Statud: _Optional[int] = ...) -> None: ...

class StartEpisode(_message.Message):
    __slots__ = ("Cycle",)
    CYCLE_FIELD_NUMBER: _ClassVar[int]
    Cycle: int
    def __init__(self, Cycle: _Optional[int] = ...) -> None: ...

class OK(_message.Message):
    __slots__ = ("OK",)
    OK_FIELD_NUMBER: _ClassVar[int]
    OK: bool
    def __init__(self, OK: bool = ...) -> None: ...
