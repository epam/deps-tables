from http import HTTPStatus

import pytest

from deps_tables.api_app import register_auth
from deps_tables.extras.auth.deps_jwt import InvalidJWTError
from tests.factories import TableFactory


@pytest.fixture
def deps_auth_service_mock(mocker, application):
    mock = mocker.Mock(application.deps_auth_service.cls)
    application.deps_auth_service.override(mock)
    yield mock

    application.deps_auth_service.reset_override()


@pytest.fixture
def register_auth_for_app(fastapi_app, application):
    application.config.authentication.enabled.override(True)
    if not getattr(fastapi_app.state, "is_auth_registered", False):
        register_auth(fastapi_app)
        fastapi_app.state.is_auth_registered = True
    yield

    application.config.authentication.enabled.override(False)


@pytest.fixture
def application_service_mock(mocker, application):
    mock = mocker.Mock(application.application.cls)
    application.application.override(mock)
    yield mock

    application.application.reset_override()


@pytest.mark.usefixtures("register_auth_for_app")
class TestDepsAuthService:
    def test_send_request_with_auth_header__token_is_valid__token_decoded_successfully__return_200(
        self, client, application_service_mock, deps_auth_service_mock, test_image_file
    ):
        token_in_request = "token"
        decoded_token = {
            "sub": "123",
            "groups": ["group1"],
            "realm_access": {"roles": []},
        }
        deps_auth_service_mock.authorize.return_value = decoded_token
        application_service_mock.detect_tables.return_value = [TableFactory()]
        response = client.post(
            url="/api/tables/v1/file/detect",
            files={
                "file": test_image_file,
            },
            headers={"Authorization": f"Bearer {token_in_request}"},
        )

        assert response.status_code == HTTPStatus.OK

    def test_send_request_with_auth_header__token_is_invalid__return_401_unauthorized(
        self, client, deps_auth_service_mock, test_image_file
    ):
        token_in_request = "token"
        deps_auth_service_mock.authorize.side_effect = InvalidJWTError
        response = client.post(
            url="/api/tables/v1/file/detect",
            files={
                "file": test_image_file,
            },
            headers={"Authorization": f"Bearer {token_in_request}"},
        )

        assert response.status_code == HTTPStatus.UNAUTHORIZED
