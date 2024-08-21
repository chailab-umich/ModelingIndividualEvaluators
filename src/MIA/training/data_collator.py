from SERDatasets import BatchCollator as SERCollator
from AnnotatorLayer import AnnotatorBatchCollator

class BatchCollator:
    def __init__(self, sample_types=None, annotator_mapper=None):
        self.main_collator = SERCollator(sample_types)
        self.batch_annotators = annotator_mapper is not None
        if annotator_mapper is not None:
            self.annotator_collator = AnnotatorBatchCollator(annotator_mapper)

    def __call__(self, samples):
        # Combine both collators into one output
        main_batch = self.main_collator(samples)
        if hasattr(self, 'annotator_collator') and self.batch_annotators:
            annotator_batch = self.annotator_collator(samples)
            main_batch = {**main_batch, **annotator_batch}
        return main_batch