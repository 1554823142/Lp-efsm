from abc import ABC, abstractmethod
from typing import Iterable, List
from core.datamodel.raw_packet import Rawpacket
from core.datamodel.session import Session
from core.datamodel.event import MessageEvent

class PCAPParser(ABC):

    @abstractmethod
    def parse(self, path: str) -> Iterable[Rawpacket]:
        pass


class SessionBuilder(ABC):

    @abstractmethod
    def build(self, packets: Iterable[Rawpacket]) -> List[Session]:
        pass



class Segmenter(ABC):

    @abstractmethod
    def segment(self, session: Session) -> List[MessageEvent]:
        pass

