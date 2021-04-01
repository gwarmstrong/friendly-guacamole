from typing import Dict, Callable
from pathlib import Path
from abc import ABC
from urllib import request
from io import BytesIO
from zipfile import ZipFile
import h5py
import os
import pandas as pd
from biom import load_table
from biom.util import biom_open
from friendly_guacamole.exceptions import DatasetError
from http.client import HTTPResponse


def download_and_extract_response(response, path, unzip=True):
    zip_path = path
    if unzip:
        os.makedirs(path, exist_ok=True)
        zip_path = path + '.zip'
    else:
        pardir = os.path.abspath(os.path.join(path, os.pardir))
        os.makedirs(pardir, exist_ok=True)

    with open(zip_path, 'wb') as fp:
        fp.write(response.read())

    if unzip:
        with ZipFile(zip_path, 'r') as fp:
            fp.extractall(path)
        os.remove(zip_path)


class ArchiverMixin:
    dataset: 'Dataset'
    save: Callable


class QiitaSaveMixin(ArchiverMixin):

    def _download_method(self, response, path, unzip=True):
        download_and_extract_response(response, path, unzip=unzip)

    def save(self, artifact: 'Artifact', response: HTTPResponse, unzip=True):
        path = os.path.join(self.dataset.path, artifact.name)
        self._download_method(response, path, unzip=unzip)


class FileSystemArchiver(QiitaSaveMixin):

    def __init__(self, dataset):
        self.dataset: Dataset = dataset
        self.path = self.dataset.path

    def read(self, artifact: 'FileSystemArtifact'):
        return artifact.filesystem_read(self)

    def exists(self, artifact: 'FileSystemArtifact'):
        return artifact.filesystem_exists(self)

    def path(self, artifact: 'FileSystemArtifact'):
        return artifact.filesystem_path(self)


class Artifact:
    name: str

    @staticmethod
    def merge(self, other):
        raise NotImplementedError()


class QiitaArtifact(Artifact):
    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        raise NotImplementedError('You should implement this.')


class FileSystemArtifact(Artifact):
    def filesystem_exists(self, archiver: FileSystemArchiver):
        raise NotImplementedError('You should implement this.')

    def filesystem_read(self, archiver: FileSystemArchiver):
        raise NotImplementedError('You should implement this.')

    def filesystem_path(self, archiver: FileSystemArchiver):
        raise NotImplementedError('You should implement this.')


class Table(QiitaArtifact, FileSystemArtifact):

    name = 'table'

    def __init__(self, artifact_id):
        self.artifact_id = artifact_id

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        table_link = self._artifact_fstring.format(self.artifact_id)
        r = client.make_request(table_link)
        archiver.save(self, r)

    def filesystem_exists(self, archiver: FileSystemArchiver):
        path = self.filesystem_path(archiver)
        return os.path.exists(path)

    def filesystem_read(self, archiver: FileSystemArchiver):
        path = self.filesystem_path(archiver)
        return load_table(path)

    def filesystem_path(self, archiver: FileSystemArchiver):
        path_to_table = os.path.join(
            archiver.path, self.name, 'BIOM',
            str(self.artifact_id),
            'otu_table.biom',
        )
        return os.path.abspath(path_to_table)

    @staticmethod
    def merge(t1, t2):
        return t1.merge(t2)


class ArtifactList(QiitaArtifact, FileSystemArtifact):

    def __init__(self, *artifacts):
        self.artifacts = list(artifacts)

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        for artifact in self.artifacts:
            artifact.qiita_download(client, archiver)

    def filesystem_exists(self, archiver: FileSystemArchiver):
        return all(
            artifact.filesystem_exists(archiver) for artifact in self.artifacts
        )

    def filesystem_read(self, archiver: FileSystemArchiver):
        if len(self.artifacts) > 0:
            artifact = self.artifacts[0]
            merged = artifact.filesystem_read(archiver)
        for artifact in self.artifacts[1:]:
            other = artifact.filesystem_read(archiver)
            merged = artifact.merge(merged, other)
        return merged


class Tables(ArtifactList):

    name = 'table'


class GenericWebArtifact(QiitaArtifact, FileSystemArtifact):

    link: str

    def __init__(self):
        self.path = None

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        r = client.make_request(self.link)
        archiver.save(self, r, unzip=False)

    def filesystem_exists(self, archiver):
        # being able to use path depends on datasets being checked for
        # existence when the dataset is instantiated
        self.path = self.filesystem_path(archiver)
        return os.path.exists(self.path)

    def filesystem_path(self, archiver):
        path = os.path.join(
            archiver.path, self.name
        )
        return os.path.abspath(path)


class GG97OTUsTree(GenericWebArtifact):

    name = '97_otus.tree'
    link = 'ftp://greengenes.microbio.me/greengenes_release/gg_13_8_otus/'\
           'trees/97_otus.tree'

    def filesystem_read(self, archiver: FileSystemArchiver):
        return self


class Metadata(QiitaArtifact, FileSystemArtifact):

    name = 'metadata'

    def __init__(self, study_id):
        self.study_id = study_id

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        metadata_link = self._metadata_fstring.format(
            self.study_id)
        r = client.make_request(metadata_link)
        archiver.save(self, r)

    def filesystem_exists(self, archiver):
        return True if self.filesystem_path(archiver) else False

    def filesystem_read(self, archiver):
        path = self.filesystem_path(archiver)
        return pd.read_csv(path, sep='\t')

    def filesystem_path(self, archiver):
        path_to_metadata = os.path.join(
            archiver.path, self.name, 'templates'
        )
        if not os.path.exists(path_to_metadata):
            return False
        md_candidates = list(
            filter(lambda n: n.endswith('.txt'),
                   os.listdir(path_to_metadata)
                   )
        )
        if len(md_candidates) == 0:
            return False
        else:
            md_file = md_candidates[0]
        full_path = os.path.join(path_to_metadata, md_file)
        return os.path.abspath(full_path)


class QiitaClient:

    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    def __init__(self, dataset, archiver: QiitaSaveMixin):
        self.dataset: Dataset = dataset
        self.archiver = archiver

    def download(self, artifact: QiitaArtifact):
        artifact.qiita_download(self, self.archiver)

    @staticmethod
    def make_request(url):
        return request.urlopen(url)


class NewDataset(ABC):

    artifacts: Dict[str, Artifact] = dict()

    def __init__(self, path, download=True):
        self.path = path

        if download:
            self.download()

        if not self._check_integrity():
            raise DatasetError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self._data = {}

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        for artifact in self.artifacts.values():
            artifact.download(self.path)

    def _check_integrity(self):
        return all(
            artifact.exists(self.path) for artifact in
            self.artifacts.values()
        )

    def __getitem__(self, item):
        if item in self._data:
            return self._data[item]
        elif item in self.artifacts:
            value = self.artifacts[item].read(self.path)
            self._data[item] = value
            return value
        else:
            raise KeyError(item)


class ArtifactInterface:

    def download(self, path):
        data = self.client.download()
        self.archiver.save(data, path)
        return data

    def read(self, path):
        data = self.archiver.read(path)
        return data

    def exists(self, path):
        return self.archiver.exists(path)


class QiitaClientInterface:
    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    @staticmethod
    def make_request(url):
        return request.urlopen(url)

    def get_data(self):
        link = self.get_link()
        zipdata = BytesIO()
        zipdata.write(self.make_request(link).read())
        myzipfile = ZipFile(zipdata)
        return myzipfile

    def download(self):
        data = self.get_data()
        return self.scavenge_data(data)


class QiitaArtifactClientInterface(QiitaClientInterface):
    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"

    def __init__(self, artifact_id):
        self.artifact_id = artifact_id

    def get_link(self):
        artifact_link = self._artifact_fstring.format(self.artifact_id)
        return artifact_link


class QiitaTableClient(QiitaArtifactClientInterface):

    def scavenge_data(self, data):
        path_to_table = os.path.join(
            'BIOM',
            str(self.artifact_id),
            'otu_table.biom',
        )
        return load_table(h5py.File(data.open(path_to_table)))


class ArchiverInterface:

    def save(self, data, path):
        raise NotImplementedError()

    def read(self, path):
        raise NotImplementedError()

    def exists(self, path):
        raise NotImplementedError()


class BIOMArchiver(ArchiverInterface):

    def __init__(self, identifier):
        self.identifier = str(identifier)

    def path(self, root):
        pardir = os.path.join(root, self.identifier, 'table')
        path = os.path.join(pardir, self.identifier +
                            '.table.biom')
        return path

    def exists(self, root):
        return os.path.exists(self.path(root))

    def read(self, root):
        path = self.path(root)
        return load_table(path)

    def save(self, data, root):
        path = self.path(root)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with biom_open(path, 'w') as fh:
            data.to_hdf5(fh, self.identifier)


class QiitaTable(ArtifactInterface):

    def __init__(self, artifact_id):
        self.archiver = BIOMArchiver(artifact_id)
        self.client = QiitaTableClient(artifact_id)


class Dataset(ABC):

    artifacts: Dict[str, Artifact] = dict()
    archiver_type = FileSystemArchiver
    client_type = QiitaClient

    def __init__(self, path, download=True):
        self.path = path
        self.archiver = self.archiver_type(self)
        self.client = self.client_type(self, self.archiver)

        if download:
            self.download()

        if not self._check_integrity():
            raise DatasetError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self._table = None
        self._metadata = None
        self._data = dict()

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        for artifact in self.artifacts.values():
            self.client.download(artifact)

    def _check_integrity(self):
        return all(
                self.archiver.exists(artifact) for artifact in
                self.artifacts.values()
            )

    def __getitem__(self, item):
        if item in self._data:
            return self._data[item]
        elif item in self.artifacts:
            value = self.archiver.read(self.artifacts[item])
            self._data[item] = value
            return value
        else:
            raise KeyError(item)


class KeyboardDataset(Dataset):
    study_id = 232
    table_artifact_id = 46809

    artifacts = {
        'metadata': Metadata(study_id),
        'table': Table(table_artifact_id),
        'tree': GG97OTUsTree(),
    }


class NewKeyboardDataset(NewDataset):
    study_id = 232
    table_artifact_id = 46809

    artifacts = {
        'table': QiitaTable(table_artifact_id),
    }


class DietInterventionStudy(Dataset):
    study_id = 11550
    table_artifact_ids = [63512, 63515]

    artifacts = {
        'metadata': Metadata(study_id),
        'table': Tables(
            Table(table_artifact_ids[0]),
            Table(table_artifact_ids[1]),
        ),
        'tree': GG97OTUsTree(),
    }
