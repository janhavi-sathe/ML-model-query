from enum import Enum
from typing import Mapping, Sequence
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_base import ExperimentPageBase
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.demo.pages import (BoxPushV2Demo, RescueDemo, RescueV2Demo,
                                       ToolDeliveryDemo, ToolHandoverDemo)


class E_SessionName(Enum):
  Movers_full_dcol = 0
  Movers_partial_dcol = 1
  Cleanup_full_dcol = 2
  Cleanup_partial_dcol = 3
  Rescue_full_dcol = 4
  Rescue_partial_dcol = 5
  Blackout_full_dcol = 6
  Blackout_partial_dcol = 7
  ToolDelivery = 8
  ToolHandover = 9


SESSION_TITLE = {
    E_SessionName.Movers_full_dcol: 'Movers - Fully Observable',
    E_SessionName.Movers_partial_dcol: 'Movers - Partially Observable',
    E_SessionName.Cleanup_full_dcol: 'Cleanup - Fully Observable',
    E_SessionName.Cleanup_partial_dcol: 'Cleanup - Partially Observable',
    E_SessionName.Rescue_full_dcol: 'Rescue - Fully Observable',
    E_SessionName.Rescue_partial_dcol: 'Rescue - Partially Observable',
    E_SessionName.Blackout_full_dcol: 'Blackout - Fully Observable',
    E_SessionName.Blackout_partial_dcol: 'Blackout - Partially Observable',
    E_SessionName.ToolDelivery: 'ToolDelivery',
    E_SessionName.ToolHandover: 'ToolHandover'
}

PAGE_LIST_MOVERS_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    BoxPushV2Demo(EDomainType.Movers, False),
]
PAGE_LIST_MOVERS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    BoxPushV2Demo(EDomainType.Movers, True),
]
PAGE_LIST_CLEANUP_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    BoxPushV2Demo(EDomainType.Cleanup, False),
]
PAGE_LIST_CLEANUP = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    BoxPushV2Demo(EDomainType.Cleanup, True),
]

PAGE_LIST_RESCUE_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    RescueDemo(False),
]

PAGE_LIST_RESCUE = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    RescueDemo(True),
]

PAGE_LIST_BLACKOUT_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Blackout),
    RescueV2Demo(False),
]

PAGE_LIST_BLACKOUT = [
    pgc.CanvasPageStart(EDomainType.Blackout),
    RescueV2Demo(True),
]

PAGE_LIST_TOOLDELIVERY = [
    pgc.CanvasPageStart(EDomainType.ToolDelivery),
    ToolDeliveryDemo(False),
]

PAGE_LIST_TOOLHANDOVER = [
    pgc.CanvasPageStart(EDomainType.ToolHandover),
    ToolHandoverDemo(False),
]

GAMEPAGES = {
    E_SessionName.Movers_full_dcol: PAGE_LIST_MOVERS_FULL_OBS,
    E_SessionName.Movers_partial_dcol: PAGE_LIST_MOVERS,
    E_SessionName.Cleanup_full_dcol: PAGE_LIST_CLEANUP_FULL_OBS,
    E_SessionName.Cleanup_partial_dcol: PAGE_LIST_CLEANUP,
    E_SessionName.Rescue_full_dcol: PAGE_LIST_RESCUE_FULL_OBS,
    E_SessionName.Rescue_partial_dcol: PAGE_LIST_RESCUE,
    E_SessionName.Blackout_full_dcol: PAGE_LIST_BLACKOUT_FULL_OBS,
    E_SessionName.Blackout_partial_dcol: PAGE_LIST_BLACKOUT,
    E_SessionName.ToolDelivery: PAGE_LIST_TOOLDELIVERY,
    E_SessionName.ToolHandover: PAGE_LIST_TOOLHANDOVER
}  # type: Mapping[E_SessionName, Sequence[ExperimentPageBase]]
