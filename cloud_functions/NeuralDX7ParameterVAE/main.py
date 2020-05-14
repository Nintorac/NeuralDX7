from agoge import InferenceWorker

worker = InferenceWorker('~/agoge/artifacts/dx7-vae/hasty-copper-dogfish_0_2020-05-06_10-46-27o654hmde/checkpoint_204/model.box', with_data=False)


def hello_get(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    return 'Hello World!'