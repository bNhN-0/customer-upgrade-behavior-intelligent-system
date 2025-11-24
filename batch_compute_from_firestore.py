!pip install flask-restful
!pip install numpy

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

from google.colab import output
output.serve_kernel_port_as_window(5000)


app = Flask(__name__)
api = Api(app)

compute_forcing_terms = {
    'forcing_term1': {'forcing_term': '0.9'},
    'forcing_term2': {'forcing_term': '0.1'},
    'forcing_term3': {'forcing_term': '0.5'},
}

def abort_if_forcing_term_doesnt_exist(forcing_term_id):
    if forcing_term_id not in compute_forcing_terms:
        abort(404, message="Forcing term {} doesn't exist".format(forcing_term_id))

parser = reqparse.RequestParser()
parser.add_argument('forcing_term')

# ForcingTerm
# shows a single forcing term item and lets you delete a forcing term item
class ForcingTerm(Resource):
    def get(self, forcing_term_id):
        abort_if_forcing_term_doesnt_exist(forcing_term_id)
        return compute_forcing_terms[forcing_term_id]

    def delete(self, forcing_term_id):
        abort_if_forcing_term_doesnt_exist(forcing_term_id)
        del compute_forcing_terms[forcing_term_id]
        return '', 204

    def put(self, forcing_term_id):
        args = parser.parse_args()
        forcing_term = {'forcing_term': args['forcing_term']}
        compute_forcing_terms[forcing_term_id] = forcing_term
        return forcing_term, 201

# ForcingTermList
# shows a list of all forcing terms, and lets you POST to compute forcing term
import numpy as np


class ForcingTermList(Resource):
    def get(self):
        return compute_forcing_terms

    def post(self):
        dl_parser = reqparse.RequestParser()
        dl_parser.add_argument('DA', type=float, required=False, default=0.9)
        dl_parser.add_argument('BH', type=float, required=False, default=1.0)
        dl_parser.add_argument('TI', type=float, required=False, default=0.0)
        dl_parser.add_argument('ENG', type=float, required=False, default=0.0)
        dl_parser.add_argument('PU', type=float, required=False, default=0.3)
        dl_parser.add_argument('SI', type=float, required=False, default=0.3)
        dl_parser.add_argument('PS', type=float, required=False, default=0.9)

        args = dl_parser.parse_args()


        # Generate a new forcing_term_id
        max_id = 0
        for key in compute_forcing_terms.keys():
            if key.startswith('forcing_term'):
                try:
                    num = int(key.lstrip('forcing_term'))
                    max_id = max(max_id, num)
                except ValueError:
                    continue
        forcing_term_id = f'forcing_term{max_id + 1}'

        # Model parameters
        dt = 0.01  # Time step
        eta = 0.9
        alpha = 0.7
        omega = 0.5
        t = 800

        # Initialize arrays
        X = np.zeros(t)
        Y = np.zeros(t)
        S = np.zeros(t)

        forcing_term = np.zeros(t)

        # Initial conditions

        X[0] = alpha * args['BH'] + (1 - alpha) * args['DA']
        Y[0] = (omega * args['DA'] + omega * args['BH']) * args['PS'] 
        S[0] = X[0] * (1 - Y[0])

        forcing_term[0] = 0.1

        # Compute forcing term over time steps
        for t_step in range(1, t):

            #H1
            #Need
            N = (args['DA'] + args['TI'] + args['ENG'] + args['PU'] + args['SI']) / 5.0
            #Bonding
            B = (args['ENG'] + args['PU'] + args['SI']) / 3.0
            #Holding Factor
            H = ((1-args['DA']) + args['BH'] + (1-args['TI']) + (1-args['ENG']) +
                (1-args['PU']) + (1-args['SI']) + args['PS']*(1-args['TI'])) / 7.0
            

            #H2
            X[t_step] = alpha*B + (1-alpha) * N - (alpha * H)  
            Y[t_step] = (omega * N + omega * B) * H   

            #Output Short term
            S[t_step] = X[t_step] * (1 - Y[t_step])

            forcing_term[t_step] = forcing_term[t_step-1] + eta * (S[t_step-1] - forcing_term[t_step-1]) * dt

        # Store the computed forcing term
        compute_forcing_terms[forcing_term_id] = {'forcing_term': str(forcing_term[-1])}

        # Return the new forcing term
        return {forcing_term_id: {'forcing_term': float(forcing_term[-1])}}, 201


api.add_resource(ForcingTermList, '/forcing_terms')
api.add_resource(ForcingTerm, '/forcing_terms/<forcing_term_id>')



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)