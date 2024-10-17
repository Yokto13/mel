from tokenization.pipeline.base import Pipeline, PipelineStep
from tokenization.pipeline.savers import NPZSaver
from tokenization.pipeline.tokenizers import SimpleTokenizer


class TestTokenizationPipeline:
    def test_add_step(self):
        pipeline = Pipeline()
        step1 = SimpleTokenizer(None, 64)
        step2 = NPZSaver("test.npz")
        pipeline.add(step1)
        pipeline.add(step2)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0] == step1
        assert pipeline.steps[1] == step2

    def test_run_pipeline(self, mocker):
        pipeline = Pipeline()
        step1 = mocker.Mock(spec=PipelineStep)
        step1.process.return_value = ["step1_output1", "step1_output2"]
        step2 = mocker.Mock(spec=PipelineStep)
        step2.process.return_value = ["step2_output1", "step2_output2"]
        pipeline.add(step1)
        pipeline.add(step2)

        pipeline.run()

        step1.process.assert_called_once_with()
        step2.process.assert_called_once_with(["step1_output1", "step1_output2"])

    def test_pipeline_str_representation(self):
        pipeline = Pipeline()
        step1 = SimpleTokenizer(None, 64)
        step2 = NPZSaver("test.npz")
        pipeline.add(step1)
        pipeline.add(step2)

        expected_str = (
            "Tokenization Pipeline Steps:\n" "1. SimpleTokenizer\n" "2. NPZSaver"
        )
        assert str(pipeline) == expected_str
