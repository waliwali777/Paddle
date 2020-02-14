# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import paddle.fluid as fluid


class Hdfs():
    def __init__():
        self.hdfs_ugi = None
        self.hdfs_name = None
        self.hdfs_path = None


class Cluster():
    def __init__(self):
        self.job_server = None
        self.pods = None
        self.hdfs = None

    def __eq__(self, cluster):
        if len(self.pods) != len(cluster.pods):
            return True

        for pod in self.pods:
            if pod != cluster.pods[i]:
                return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(cluster):
        self.pods = copy.copy(cluster.pods)
        pass

    def world_rank(self):
        count = 0
        for pod in self.pods:
            for gpu in pod.gpus:
                count += 1
        return count

    def get_trainer_endpoints(self):
        pass

    def get_pods_endpints(self):
        pass


class JobServer():
    def __init__(self):
        self.endpoint = None


class Trainer():
    def __init__(self):
        self.gpu = []
        self.endpoint = []
        self.rank = []

    def __eq__(self):
        pass

    def __ne__(self):
        pass

    def ranks(self):
        return self.rank


class Pod():
    def __init__(self):
        self.idx = None
        self.ip = None
        self.port = None
        self.trainers = []

    def __eq__(self, pod):
        pass

    def __ne__(self, pod):
        return not self == pod

    def parse_response(self, res_pods):
        pass

    def ranks(self):
        r = []
        for t in self.trainers:
            r.extend(t.ranks)

        return r


class Gloo():
    def __init__():
        #self.__iface = self.__get_default_iface()
        self._prefix = "edl_job"
        self._gloo = fluid.core.Gloo()

        self._endpoints = None
        self._hdfs = None
        self._rank = None

    def _init(hdfs, endpints, rank, try_num=3):
        if endpoints != self._endpoints or hdfs != self._hdfs or rank != self._rank:
            self._hdfs = hdfs
            self._endpoints = self._endpoints
            self._rank = rank
            self._try_num = try_num

            iface = self.__get_default_iface()
            return self.__gloo.init(pod.idx,
                                    len(pods_endpoints),
                                    hdfs.hdfs_path.rstrip("/") + "/all",
                                    hdfs.hdfs_name, hdfs.hdfs_ugi, self.__iface,
                                    self._prefix)

        return True

    def _loop(func):
        while True:
            if func():
                return True

            if step > self._try_num:
                break

            time.sleep(3)
            step += 1

        return False

    def init(hdfs, endpoints, rank, try_num=3):
        func = functools.partial(
            self._init,
            hdfs=hdfs,
            endpoints=endpoints,
            rank=rank,
            try_num=try_num)
        return self._loop(func)

    def barrier(timeout):
        func = functools.partial(self._gloo.barrier, timeout=timeout)
        return self._loop(func)

    def __get_default_iface(self):
        """
        get default physical interface
        """
        default1 = self.__get_default_iface_from_gateway()
        default2 = self.__get_default_iface_from_interfaces()
        return default2 if default1 == "lo" else default1

    def __get_default_iface_from_gateway(self):
        """
        get default physical interface
        """
        import netifaces
        gateways = netifaces.gateways()
        if gateways.get(netifaces.AF_INET) != None:
            gateway = gateways[netifaces.AF_INET]
            if len(gateway) > 0 and len(gateway[0]) > 1:
                return gateway[0][1]
        return "lo"

    def __get_default_iface_from_interfaces(self):
        """
        get default physical interface
        """
        import netifaces
        for intf_name in netifaces.interfaces():
            addresses = netifaces.ifaddresses(intf_name)
            if netifaces.AF_INET in addresses:
                ipv4_addresses = addresses[netifaces.AF_INET]
                for ipv4_address in ipv4_addresses:
                    if 'broadcast' in ipv4_address:
                        return intf_name
        return "lo"
