import sys
sys.path.append('OpenNMT')



import OpenNMT
import utils.tensor_utils as tu



class CustomModel(OpenNMT.onmt.models.model.NMTModel):

    def forward(self,_x,_y):
        '''
            Explicitly passes it through generator
            I don't know why
        '''

        h_unsort, o_unsort, lengths = self.encoder(_x.transpose(1,0).unsqueeze(-1))
        self.decoder.init_state(_x.transpose(1,0).unsqueeze(-1), o_unsort, h_unsort)

        dec_out, attns = self.decoder(_y.transpose(1,0).unsqueeze(-1), o_unsort,
                                  memory_lengths=lengths)
        dec_out = dec_out.view(-1, dec_out.size(2))
        generated_output = self.generator(dec_out)

        return generated_output