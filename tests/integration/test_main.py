import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from src.main import main
import pytest
from unittest.mock import patch

class TestMainIntegration(unittest.TestCase):
    @patch('pipelines.train.train_wrapper')
    @patch('pipelines.eval.eval_wrapper')
    @patch('utils.config_loader.load_config')
    def test_train_pipeline(self, mock_load_config, mock_eval_wrapper, mock_train_wrapper):
        dummy_config = {'dummy': 'config'}
        mock_load_config.return_value = dummy_config

        args = SimpleNamespace(config='configs/templates/default_train.yaml', pipeline='train')
        @pytest.fixture
        def dummy_config():
            return {'dummy': 'config'}

        @patch('pipelines.train.train_wrapper')
        @patch('pipelines.eval.eval_wrapper')
        @patch('utils.config_loader.load_config')
        def test_train_pipeline(mock_load_config, mock_eval_wrapper, mock_train_wrapper, dummy_config):
            mock_load_config.return_value = dummy_config
            args = SimpleNamespace(config='configs/templates/default_train.yaml', pipeline='train')
            main(args)
            mock_load_config.assert_called_once_with('configs/templates/default_train.yaml')
            mock_train_wrapper.assert_called_once_with(dummy_config)
            mock_eval_wrapper.assert_not_called()

        @patch('pipelines.train.train_wrapper')
        @patch('pipelines.eval.eval_wrapper')
        @patch('utils.config_loader.load_config')
        def test_eval_pipeline(mock_load_config, mock_eval_wrapper, mock_train_wrapper, dummy_config):
            mock_load_config.return_value = dummy_config
            args = SimpleNamespace(config='configs/templates/default_train.yaml', pipeline='eval')
            main(args)
            mock_load_config.assert_called_once_with('configs/templates/default_train.yaml')
            mock_eval_wrapper.assert_called_once_with(dummy_config)
            mock_train_wrapper.assert_not_called()

        @patch('utils.config_loader.load_config')
        def test_unsupported_pipeline(mock_load_config, dummy_config):
            mock_load_config.return_value = dummy_config
            args = SimpleNamespace(config='configs/templates/default_train.yaml', pipeline='invalid')
            with pytest.raises(ValueError, match='Unsupported pipeline: invalid'):
                main(args)

        @patch('pipelines.train.train_wrapper')
        @patch('utils.config_loader.load_config')
        def test_train_pipeline_with_different_config(mock_load_config, mock_train_wrapper, dummy_config):
            mock_load_config.return_value = dummy_config
            args = SimpleNamespace(config='configs/other_train.yaml', pipeline='train')
            main(args)
            mock_load_config.assert_called_once_with('configs/other_train.yaml')
            mock_train_wrapper.assert_called_once_with(dummy_config)

        @patch('pipelines.eval.eval_wrapper')
        @patch('utils.config_loader.load_config')
        def test_eval_pipeline_with_different_config(mock_load_config, mock_eval_wrapper, dummy_config):
            mock_load_config.return_value = dummy_config
            args = SimpleNamespace(config='configs/other_eval.yaml', pipeline='eval')
            main(args)
            mock_load_config.assert_called_once_with('configs/other_eval.yaml')
            mock_eval_wrapper.assert_called_once_with(dummy_config)

        @patch('utils.config_loader.load_config')
        def test_missing_config_argument(mock_load_config):
            args = SimpleNamespace(pipeline='train')
            with pytest.raises(AttributeError):
                main(args)

        @patch('utils.config_loader.load_config')
        def test_missing_pipeline_argument(mock_load_config):
            args = SimpleNamespace(config='configs/templates/default_train.yaml')
            with pytest.raises(AttributeError):
                main(args)