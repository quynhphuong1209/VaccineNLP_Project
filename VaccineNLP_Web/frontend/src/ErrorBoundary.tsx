import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(_: Error): State {
    return { hasError: true };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error in ErrorBoundary:", error, errorInfo);
  }

  private handleReload = () => {
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '24px',
          textAlign: 'center',
          backgroundColor: '#0f172a',
          color: '#f8f9fa',
          fontFamily: 'system-ui, -apple-system, sans-serif',
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px'
        }}>
          <div style={{ fontSize: '32px' }}>⚠️</div>
          <h2 style={{ fontSize: '18px', fontWeight: '600', color: '#f8f9fa', margin: '0' }}>
            Giao diện gặp lỗi hiển thị — kết quả phân loại vẫn an toàn.
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '14px', margin: '0 0 12px 0' }}>
            Vui lòng nhấn nút dưới đây để tải lại trang.
          </p>
          <button 
            onClick={this.handleReload}
            style={{
              padding: '10px 20px',
              backgroundColor: '#0f766e',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '14px',
              transition: 'background-color 0.2s'
            }}
          >
            Tải lại trang
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
